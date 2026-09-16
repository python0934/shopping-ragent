"""
User service — CRUD, page query, change password.

Mirrors Java UserServiceImpl.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ClientException
from app.core.snowflake import get_snowflake_id_str
from app.core.user_context import UserContext
from app.models.system import UserDO
from app.schemas.user import (
    ChangePasswordRequest,
    CurrentUserVO,
    UserCreateRequest,
    UserPageRequest,
    UserUpdateRequest,
    UserVO,
)

logger = logging.getLogger(__name__)

DEFAULT_ADMIN_USERNAME = "admin"
VALID_ROLES = {"admin", "user"}


class UserService:
    """用户服务 — CRUD、分页查询、修改密码"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def page_query(self, request: UserPageRequest) -> dict[str, Any]:
        """用户分页查询"""
        keyword = (request.keyword or "").strip() or None

        query = select(UserDO).where(UserDO.deleted == 0)
        count_query = select(func.count()).select_from(UserDO).where(UserDO.deleted == 0)

        if keyword:
            like_pattern = f"%{keyword}%"
            filter_cond = or_(
                UserDO.username.like(like_pattern),
                UserDO.role.like(like_pattern),
            )
            query = query.where(filter_cond)
            count_query = count_query.where(filter_cond)

        query = query.order_by(UserDO.update_time.desc())
        query = query.offset((request.current - 1) * request.size).limit(request.size)

        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        result = await self.db.execute(query)
        records = result.scalars().all()

        return {
            "records": [self._to_vo(r) for r in records],
            "total": total,
            "current": request.current,
            "size": request.size,
        }

    async def create(self, request: UserCreateRequest) -> str:
        """创建用户"""
        username = (request.username or "").strip()
        password = (request.password or "").strip()
        if not username:
            raise ClientException("用户名不能为空")
        if not password:
            raise ClientException("密码不能为空")
        if username.lower() == DEFAULT_ADMIN_USERNAME:
            raise ClientException("默认管理员用户名不可用")

        role = self._normalize_role(request.role)
        await self._ensure_username_available(username, exclude_id=None)

        user_id = get_snowflake_id_str()
        record = UserDO(
            id=user_id,
            username=username,
            password=password,
            role=role,
            avatar=(request.avatar or "").strip() or None,
        )
        self.db.add(record)
        await self.db.flush()
        return user_id

    async def update(self, user_id: str, request: UserUpdateRequest) -> None:
        """更新用户"""
        record = await self._load_by_id(user_id)
        self._ensure_not_default_admin(record)

        if request.username is not None:
            username = request.username.strip()
            if not username:
                raise ClientException("用户名不能为空")
            if username != record.username:
                if username.lower() == DEFAULT_ADMIN_USERNAME:
                    raise ClientException("默认管理员用户名不可用")
                await self._ensure_username_available(username, exclude_id=user_id)
            record.username = username

        if request.role is not None:
            record.role = self._normalize_role(request.role)

        if request.avatar is not None:
            record.avatar = request.avatar.strip() or None

        if request.password is not None:
            password = request.password.strip()
            if not password:
                raise ClientException("新密码不能为空")
            record.password = password

        await self.db.flush()

    async def delete(self, user_id: str) -> None:
        """删除用户（软删除）"""
        record = await self._load_by_id(user_id)
        self._ensure_not_default_admin(record)
        record.deleted = 1
        await self.db.flush()

    async def change_password(self, request: ChangePasswordRequest) -> None:
        """修改当前用户密码"""
        current_pwd = (request.currentPassword or "").strip()
        new_pwd = (request.newPassword or "").strip()
        if not current_pwd:
            raise ClientException("当前密码不能为空")
        if not new_pwd:
            raise ClientException("新密码不能为空")

        login_user = UserContext.require_user()
        result = await self.db.execute(
            select(UserDO).where(UserDO.id == login_user.userId, UserDO.deleted == 0)
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise ClientException("用户不存在")
        if record.password != current_pwd:
            raise ClientException("当前密码不正确")
        record.password = new_pwd
        await self.db.flush()

    async def get_current_user(self) -> CurrentUserVO:
        """获取当前登录用户信息"""
        user = UserContext.require_user()
        return CurrentUserVO(
            userId=user.userId,
            username=user.username,
            role=user.role,
            avatar=user.avatar,
        )

    async def _load_by_id(self, user_id: str) -> UserDO:
        result = await self.db.execute(
            select(UserDO).where(UserDO.id == user_id, UserDO.deleted == 0)
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise ClientException("用户不存在")
        return record

    @staticmethod
    def _ensure_not_default_admin(record: UserDO) -> None:
        if record and record.username and record.username.lower() == DEFAULT_ADMIN_USERNAME:
            raise ClientException("默认管理员不允许修改或删除")

    async def _ensure_username_available(self, username: str, exclude_id: str | None) -> None:
        query = select(UserDO).where(UserDO.username == username, UserDO.deleted == 0)
        if exclude_id:
            query = query.where(UserDO.id != exclude_id)
        result = await self.db.execute(query)
        if result.scalar_one_or_none() is not None:
            raise ClientException("用户名已存在")

    @staticmethod
    def _normalize_role(role: str | None) -> str:
        value = (role or "").strip()
        if not value:
            return "user"
        if value.lower() in VALID_ROLES:
            return value.lower()
        raise ClientException("角色类型不合法")

    @staticmethod
    def _to_vo(record: UserDO) -> UserVO:
        return UserVO(
            id=record.id,
            username=record.username or "",
            role=record.role or "user",
            avatar=record.avatar or "",
            createTime=record.create_time,
            updateTime=record.update_time,
        )

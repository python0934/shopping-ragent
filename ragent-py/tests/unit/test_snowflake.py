"""Unit tests for app.core.snowflake — Snowflake ID generator."""

from app.core.snowflake import SnowflakeGenerator, get_snowflake_id, get_snowflake_id_str


class TestSnowflakeGenerator:
    def test_id_is_positive_int(self):
        gen = SnowflakeGenerator(worker_id=1)
        id_val = gen.next_id()
        assert isinstance(id_val, int)
        assert id_val > 0

    def test_id_str_is_19_digits(self):
        gen = SnowflakeGenerator(worker_id=1)
        id_str = gen.next_id_str()
        assert isinstance(id_str, str)
        assert len(id_str) == 19
        assert id_str.isdigit()

    def test_ids_are_monotonically_increasing(self):
        gen = SnowflakeGenerator(worker_id=1)
        ids = [gen.next_id() for _ in range(100)]
        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1]

    def test_ids_are_unique(self):
        gen = SnowflakeGenerator(worker_id=1)
        ids = {gen.next_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_different_worker_ids_produce_different_ids(self):
        gen1 = SnowflakeGenerator(worker_id=1)
        gen2 = SnowflakeGenerator(worker_id=2)
        id1 = gen1.next_id()
        id2 = gen2.next_id()
        assert id1 != id2

    def test_invalid_worker_id_raises(self):
        try:
            SnowflakeGenerator(worker_id=-1)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_worker_id_too_large_raises(self):
        try:
            SnowflakeGenerator(worker_id=1024)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_module_level_functions(self):
        id_int = get_snowflake_id()
        assert isinstance(id_int, int)
        assert id_int > 0

        id_str = get_snowflake_id_str()
        assert isinstance(id_str, str)
        assert len(id_str) == 19

    def test_compatibility_with_java_snowflake(self):
        """
        Java Hutool snowflake produces 19-digit IDs.
        Our Python implementation must also produce 19-digit IDs.
        """
        gen = SnowflakeGenerator(worker_id=1)
        for _ in range(50):
            id_str = gen.next_id_str()
            assert len(id_str) == 19, f"Expected 19 digits, got {len(id_str)}: {id_str}"
            # Must be parseable as a long (Java long max = 19 digits)
            assert int(id_str) < 2**63

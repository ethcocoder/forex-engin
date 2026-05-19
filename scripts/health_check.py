import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.loader import load_config


def test_timescaledb(config) -> bool:
    print("Testing TimescaleDB (PostgreSQL) connection...")
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=config.database.timescaledb.host,
            port=config.database.timescaledb.port,
            user=config.database.timescaledb.username,
            password=config.database.timescaledb.password,
            database=config.database.timescaledb.database,
            connect_timeout=3,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION();")
        db_version = cursor.fetchone()
        print(f"  [SUCCESS] Connected to TimescaleDB. Version: {db_version[0]}")
        cursor.close()
        conn.close()
        return True
    except ImportError:
        print("  [WARNING] 'psycopg2' not installed. Skipping direct DB check.")
        return True
    except Exception as e:
        print(f"  [ERROR] TimescaleDB connection failed: {e}")
        return False


def test_redis(config) -> bool:
    print("Testing Redis connection...")
    try:
        import redis

        r = redis.Redis(
            host=config.database.redis.host,
            port=config.database.redis.port,
            password=config.database.redis.password,
            db=config.database.redis.db,
            socket_timeout=3,
        )
        ping_ok = r.ping()
        if ping_ok:
            print("  [SUCCESS] Connected to Redis (Ping OK).")
            return True
        else:
            print("  [ERROR] Redis ping returned unexpected result.")
            return False
    except ImportError:
        print("  [WARNING] 'redis' not installed. Skipping direct Redis check.")
        return True
    except Exception as e:
        print(f"  [ERROR] Redis connection failed: {e}")
        return False


def test_kafka(config) -> bool:
    print("Testing Kafka brokers connection...")
    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=config.kafka.bootstrap_servers,
            request_timeout_ms=3000,
        )
        print("  [SUCCESS] Connected to Kafka Brokers.")
        producer.close()
        return True
    except ImportError:
        print("  [WARNING] 'kafka-python' not installed. Skipping direct Kafka check.")
        return True
    except Exception as e:
        print(f"  [ERROR] Kafka connection failed: {e}")
        return False


def main() -> None:
    print("=========================================")
    print(" FOREX NEURAL ENGINE SYSTEM HEALTH CHECK ")
    print("=========================================")

    # Load local configuration
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "config.yaml",
    )
    try:
        config = load_config(config_path)
        print(f"[SUCCESS] Configuration loaded from {config_path}")
        print(f"  Environment: {config.environment}")
        print(f"  Target Pairs: {config.pairs}")
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    print("-----------------------------------------")

    # Run system check evaluations
    db_ok = test_timescaledb(config)
    redis_ok = test_redis(config)
    kafka_ok = test_kafka(config)

    print("-----------------------------------------")
    if db_ok and redis_ok and kafka_ok:
        print("[SUCCESS] All systems verified or warnings handled gracefully. Dev ready.")
        sys.exit(0)
    else:
        print("[CRITICAL] Health check failed for one or more core infrastructure services.")
        sys.exit(1)


if __name__ == "__main__":
    main()

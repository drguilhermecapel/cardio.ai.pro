# Substitua a função verify_password em backend/app/core/security.py por:

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        # Invalid hash format
        logger.warning("Invalid bcrypt hash format")
        return False
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

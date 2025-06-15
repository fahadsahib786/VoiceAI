from datetime import datetime, timedelta
from typing import Optional, List
import bcrypt
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

# Create SQLite database engine
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    subscription_end_date = Column(DateTime)
    monthly_char_limit = Column(Integer, default=1000000)  # 1M characters default
    chars_used_today = Column(Integer, default=0)
    last_usage_date = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class DBManager:
    def get_db(self) -> Session:
        db = SessionLocal()
        try:
            return db
        finally:
            db.close()

    def create_tables(self):
        Base.metadata.create_all(bind=engine)

    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

    def create_user(self, db: Session, email: str, password: str, is_admin: bool = False,
                   subscription_months: int = 1, monthly_char_limit: int = 1000000) -> User:
        hashed_password = self.hash_password(password)
        subscription_end = datetime.now() + timedelta(days=30 * subscription_months)
        
        user = User(
            email=email,
            password_hash=hashed_password,
            is_admin=is_admin,
            is_active=True,
            subscription_end_date=subscription_end,
            monthly_char_limit=monthly_char_limit,
            chars_used_today=0,
            last_usage_date=datetime.now()
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def get_user_by_id(self, db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    def get_all_users(self, db: Session) -> List[User]:
        return db.query(User).all()

    def update_user(self, db: Session, user_id: int, **kwargs) -> Optional[User]:
        user = self.get_user_by_id(db, user_id)
        if not user:
            return None
            
        for key, value in kwargs.items():
            if key == 'password':
                value = self.hash_password(value)
                key = 'password_hash'
            setattr(user, key, value)
            
        db.commit()
        db.refresh(user)
        return user

    def delete_user(self, db: Session, user_id: int) -> bool:
        user = self.get_user_by_id(db, user_id)
        if not user:
            return False
            
        db.delete(user)
        db.commit()
        return True

    def check_user_limits(self, db: Session, user_id: int, char_count: int) -> tuple[bool, str]:
        user = self.get_user_by_id(db, user_id)
        if not user:
            return False, "User not found"
            
        if not user.is_active:
            return False, "User account is suspended"
            
        if user.subscription_end_date < datetime.now():
            return False, "Subscription has expired"

        # Reset daily usage if it's a new day
        if user.last_usage_date.date() != datetime.now().date():
            user.chars_used_today = 0
            user.last_usage_date = datetime.now()

        # Check character limits
        if char_count > 1000000:  # 1M characters per request limit
            return False, "Request exceeds maximum character limit of 1M"
            
        if user.chars_used_today + char_count > 10000000:  # 10M characters per day limit
            return False, "Daily character limit exceeded"

        return True, "OK"

    def update_usage(self, db: Session, user_id: int, char_count: int):
        user = self.get_user_by_id(db, user_id)
        if not user:
            return
            
        # Reset if it's a new day
        if user.last_usage_date.date() != datetime.now().date():
            user.chars_used_today = 0
            
        user.chars_used_today += char_count
        user.last_usage_date = datetime.now()
        db.commit()

# Create global instance
db_manager = DBManager()

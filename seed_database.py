#!/usr/bin/env python3
"""
Database seeding script for TTS Server
Creates a permanent admin user and 10 test users with 1-month plans
"""

import sys
from datetime import datetime, timedelta
from database import db_manager, SessionLocal

def seed_database():
    """Create initial admin and test users"""
    db = SessionLocal()
    
    try:
        # Create database tables
        print("Creating database tables...")
        db_manager.create_tables()
        
        # Check if admin already exists
        existing_admin = db_manager.get_user_by_email(db, "admin@tts.local")
        if existing_admin:
            print("Admin user already exists. Skipping admin creation.")
        else:
            # Create permanent admin user
            print("Creating permanent admin user...")
            admin_user = db_manager.create_user(
                db=db,
                email="admin@tts.local",
                password="admin123",
                is_admin=True,
                subscription_months=999999,  # Permanent subscription
                monthly_char_limit=1000000000  # 1B characters for admin
            )
            print(f"✅ Admin user created: {admin_user.email}")
        
        # Create 10 test users with different character limits (all 1-month plans)
        test_users = [
            {"email": "user1@test.com", "char_limit": 100000},    # 100K
            {"email": "user2@test.com", "char_limit": 250000},    # 250K
            {"email": "user3@test.com", "char_limit": 500000},    # 500K
            {"email": "user4@test.com", "char_limit": 750000},    # 750K
            {"email": "user5@test.com", "char_limit": 1000000},   # 1M
            {"email": "user6@test.com", "char_limit": 2000000},   # 2M
            {"email": "user7@test.com", "char_limit": 3000000},   # 3M
            {"email": "user8@test.com", "char_limit": 4000000},   # 4M
            {"email": "user9@test.com", "char_limit": 5000000},   # 5M
            {"email": "user10@test.com", "char_limit": 10000000}, # 10M
        ]
        
        print("\nCreating test users...")
        created_count = 0
        for user_data in test_users:
            # Check if user already exists
            existing_user = db_manager.get_user_by_email(db, user_data["email"])
            if existing_user:
                print(f"⚠️  User {user_data['email']} already exists. Skipping.")
                continue
                
            user = db_manager.create_user(
                db=db,
                email=user_data["email"],
                password="password123",
                is_admin=False,
                subscription_months=1,  # 1-month plan
                monthly_char_limit=user_data["char_limit"]
            )
            created_count += 1
            print(f"✅ Created user: {user.email} (1-month plan, {user_data['char_limit']:,} chars/month)")
        
        print(f"\n🎉 Database seeding completed!")
        print(f"\n🔑 Login Credentials:")
        print(f"   Admin: admin@tts.local / admin123 (permanent)")
        print(f"   Test Users: user1@test.com to user10@test.com / password123 (1-month plans)")
        
        print(f"\n📋 Admin Panel Features:")
        print(f"   ✅ Create new users with custom limits")
        print(f"   ✅ Update user email, password, character limits")
        print(f"   ✅ Extend/modify subscription periods")
        print(f"   ✅ Suspend/activate users")
        print(f"   ✅ Delete users")
        print(f"   ✅ View detailed user information and usage")
        
        print(f"\n📊 Character Limits (Monthly):")
        for i, user_data in enumerate(test_users, 1):
            print(f"   User {i:2d}: {user_data['char_limit']:>10,} chars/month")
        print(f"   Admin:   {1000000000:>10,} chars/month")
        
        print(f"\n⚠️  Platform Limits (Applied by Server):")
        print(f"   Per Request: 1,000,000 characters maximum")
        print(f"   Per Day: 10,000,000 characters maximum")
        print(f"   Monthly: Based on user's subscription plan")
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        db.rollback()
        return False
    finally:
        db.close()
    
    return True

if __name__ == "__main__":
    print("🌱 TTS Server Database Seeding")
    print("=" * 40)
    
    success = seed_database()
    
    if success:
        print("\n✅ Database seeding completed successfully!")
        print("\n🚀 Next Steps:")
        print("   1. Run: python server.py")
        print("   2. Visit: http://localhost:8000/docs (API documentation)")
        print("   3. Login as admin to manage users")
        sys.exit(0)
    else:
        print("\n❌ Database seeding failed!")
        sys.exit(1)

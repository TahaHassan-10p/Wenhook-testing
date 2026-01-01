#!/usr/bin/env python3
"""
Script to clean up repository config ID 5 only for testing
"""

from app.database.session import SessionLocal
from app.database.models import RepositoryConfig, PRAnalysis, TestCaseDelta, IntegrationLog

def cleanup_config_5():
    """Delete repository config ID 5 and its related data"""
    
    db = SessionLocal()
    
    try:
        print("=" * 80)
        print("CLEANING UP REPOSITORY CONFIG ID 5 (GITHUB)")
        print("=" * 80)
        
        config_id = 5
        repo_config = db.query(RepositoryConfig).filter(RepositoryConfig.id == config_id).first()
        
        if not repo_config:
            print(f"❌ Config ID {config_id} not found")
            return
            
        print(f"🗑️  DELETING Config ID {config_id}:")
        print(f"   Repository: {repo_config.repository_url}")
        print(f"   Project ID: {repo_config.project_id}")
        print(f"   Token (first 20 chars): {repo_config.access_token[:20]}...")
        
        # Delete related data in correct order (foreign key constraints)
        
        # 1. Delete test case deltas
        deltas_count = db.query(TestCaseDelta).join(PRAnalysis).filter(
            PRAnalysis.repository_config_id == config_id
        ).count()
        if deltas_count > 0:
            db.query(TestCaseDelta).join(PRAnalysis).filter(
                PRAnalysis.repository_config_id == config_id
            ).delete(synchronize_session=False)
            print(f"   ✅ Deleted {deltas_count} test case deltas")
        
        # 2. Delete PR analyses
        analyses_count = db.query(PRAnalysis).filter(
            PRAnalysis.repository_config_id == config_id
        ).count()
        if analyses_count > 0:
            db.query(PRAnalysis).filter(
                PRAnalysis.repository_config_id == config_id
            ).delete(synchronize_session=False)
            print(f"   ✅ Deleted {analyses_count} PR analyses")
        
        # 3. Delete integration logs
        logs_count = db.query(IntegrationLog).filter(
            IntegrationLog.repository_config_id == config_id
        ).count()
        if logs_count > 0:
            db.query(IntegrationLog).filter(
                IntegrationLog.repository_config_id == config_id
            ).delete(synchronize_session=False)
            print(f"   ✅ Deleted {logs_count} integration logs")
        
        # 4. Finally delete the repository config
        db.delete(repo_config)
        print(f"   ✅ Deleted repository config")
        
        db.commit()
        print(f"\n✅ CONFIG 5 CLEANUP COMPLETE!")
        print(f"\nNext steps:")
        print(f"1. Go to frontend and reconnect the GitHub repository")
        print(f"2. Add your access token - it will be properly encrypted now")
        print(f"3. Check the webhook URL for the new config_id")
        print(f"4. Test that the encryption error is gone")
        
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    cleanup_config_5()

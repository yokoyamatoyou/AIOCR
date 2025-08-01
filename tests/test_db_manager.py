from core.db_manager import DBManager


def test_db_manager(tmp_path):
    db_file = tmp_path / "test.db"
    db = DBManager(db_path=str(db_file))
    db.initialize()

    job_id = db.create_job("invoice", "2025-01-01T00:00:00")
    result_id = db.add_result(job_id, "img.png", "zip_code", final_text="OLD")

    db.update_result(result_id, "NEW")
    results = list(db.fetch_results(job_id))
    assert len(results) == 1
    assert results[0]["final_text"] == "NEW"
    assert results[0]["corrected_by_user"] == 1
    # default status should be set to "confirmed" when updating
    assert results[0]["status"] == "confirmed"

    db.close()

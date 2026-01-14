import json
import os

class KBManager:
    def __init__(self, file_path='kbs.json'):
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"kbs": []}, f, ensure_ascii=False, indent=4)

    def load_kbs(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                kbs = data.get("kbs", [])
                for kb in kbs:
                    for key in ['name', 'kb_id', 'ds_id', 'bucket', 'prefix']:
                        if key in kb and isinstance(kb[key], str):
                            kb[key] = kb[key].strip()
                return kbs
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def save_kb(self, name, kb_id, ds_id, bucket, prefix):
        kbs = self.load_kbs()
        # 중복 체크 (KB ID 기준)
        if any(kb['kb_id'] == kb_id for kb in kbs):
            return False, "이미 존재하는 Knowledge Base ID입니다."
        
        name = name.strip()
        kb_id = kb_id.strip()
        ds_id = ds_id.strip()
        bucket = bucket.strip()
        prefix = prefix.strip().strip('/')
        if prefix:
            prefix += '/'

        new_kb = {
            "name": name,
            "kb_id": kb_id,
            "ds_id": ds_id,
            "bucket": bucket,
            "prefix": prefix
        }
        kbs.append(new_kb)
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump({"kbs": kbs}, f, ensure_ascii=False, indent=4)
        return True, "성공적으로 등록되었습니다."

    def delete_kb(self, kb_id):
        kbs = self.load_kbs()
        new_kbs = [kb for kb in kbs if kb['kb_id'] != kb_id]
        
        if len(kbs) == len(new_kbs):
            return False, "해당 ID를 찾을 수 없습니다."
            
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump({"kbs": new_kbs}, f, ensure_ascii=False, indent=4)
        return True, "삭제되었습니다."

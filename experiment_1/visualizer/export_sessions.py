import re
import json
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sessions_dir = current_dir.parent / "sessions"
output_json = current_dir / "session_data.json"

def parse_session_file(file_path: Path, status: str) -> list[dict]:
    if not file_path.exists():
        return []
        
    print(f"Parsing {file_path.name}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split content by the separator lines
    sessions_raw = content.split("================================================================================")
    
    parsed_sessions = []
    for raw in sessions_raw:
        raw = raw.strip()
        if not raw or "SESSION:" not in raw:
            continue
            
        # Parse Session ID and Scenario
        meta_match = re.search(r"(SUCCESSFUL|FAILED) SESSION:\s*(\w+)\s*\|\s*Scenario:\s*(\w+)", raw)
        if not meta_match:
            continue
            
        is_success = meta_match.group(1) == "SUCCESSFUL"
        session_id = meta_match.group(2)
        scenario = meta_match.group(3)
        
        # Parse Target Product Brand and Title
        target_product_match = re.search(r"Target Product:\s*\[([^\]]+)\]\s*(.*)", raw)
        target_brand = target_product_match.group(1).strip() if target_product_match else "Unknown"
        target_title = target_product_match.group(2).strip() if target_product_match else "Unknown"
        
        # Parse Target ASIN
        asin_match = re.search(r"Target ASIN:\s*(\w+)", raw)
        target_asin = asin_match.group(1).strip() if asin_match else ""
        
        # Parse Constraints
        hard_match = re.search(r"Hard Constraints:\s*(.*)", raw)
        soft_match = re.search(r"Soft Preferences:\s*(.*)", raw)
        hard_constraints = hard_match.group(1).strip() if hard_match else "[]"
        soft_preferences = soft_match.group(1).strip() if soft_match else "[]"
        
        # Parse Turns
        turns = []
        turn_blocks = raw.split("[Turn ")
        for block in turn_blocks[1:]:
            block = block.strip()
            # Extract turn number
            turn_num_match = re.match(r"(\d+)\]", block)
            if not turn_num_match:
                continue
            turn_num = int(turn_num_match.group(1))
            
            # Extract Customer message
            cust_match = re.search(r"Customer:\s*\"([^\"]+)\"", block)
            customer_msg = cust_match.group(1).strip() if cust_match else ""
            
            # Extract Copilot message
            copilot_msg_match = re.search(r"Copilot:\s*\"([^\"]+)\"", block)
            copilot_msg = copilot_msg_match.group(1).strip() if copilot_msg_match else ""
            
            # Extract Copilot requested attribute
            copilot_ask_match = re.search(r"Copilot requested attribute:\s*(\w+)", block)
            copilot_ask = copilot_ask_match.group(1).strip() if copilot_ask_match else "other"
            
            # Extract recommendations
            recommendations = []
            recs_section = block.split("Copilot Top Recommendations:")
            if len(recs_section) > 1:
                recs_lines = recs_section[1].strip().split("\n")
                for line in recs_lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Match: 1. [Brand] Title... <--- (TARGET WAS RECOMMENDED!) or ASIN
                    rec_match = re.match(r"(\d+)\.\s*\[([^\]]+)\]\s*(.*?)(?:\s*(<---.*))?$", line)
                    if rec_match:
                        rank = int(rec_match.group(1))
                        rec_brand = rec_match.group(2).strip()
                        rec_title = rec_match.group(3).strip()
                        is_target = rec_match.group(4) is not None
                        
                        recommendations.append({
                            "rank": rank,
                            "brand": rec_brand,
                            "title": rec_title,
                            "is_target": is_target
                        })
            
            turns.append({
                "turn_num": turn_num,
                "customer_msg": customer_msg,
                "copilot_msg": copilot_msg,
                "copilot_ask": copilot_ask,
                "recommendations": recommendations
            })
            
        parsed_sessions.append({
            "session_id": session_id,
            "scenario": scenario,
            "status": "success" if is_success else "failed",
            "target_asin": target_asin,
            "target_brand": target_brand,
            "target_title": target_title,
            "hard_constraints": hard_constraints,
            "soft_preferences": soft_preferences,
            "turns": turns
        })
        
    return parsed_sessions

def main():
    if not sessions_dir.exists():
        print(f"Sessions log directory not found at: {sessions_dir}")
        return
        
    all_sessions = []
    
    # Parse failed sessions
    all_sessions.extend(parse_session_file(sessions_dir / "failed_sessions.txt", "failed"))
    
    # Parse successful sessions
    for path in sessions_dir.glob("successful_*.txt"):
        all_sessions.extend(parse_session_file(path, "success"))
        
    print(f"Total parsed sessions: {len(all_sessions)}")
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2)
        
    print(f"Saved session data to: {output_json}")

if __name__ == "__main__":
    main()

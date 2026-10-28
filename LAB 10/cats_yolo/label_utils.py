from pathlib import Path


def validate_label(path: Path):
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"line {line_number}: ต้องมี 5 ค่า")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: มีค่าที่ไม่ใช่ตัวเลข")
            continue
        if class_id != 0:
            errors.append(f"line {line_number}: class_id ต้องเป็น 0")
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            errors.append(f"line {line_number}: พิกัดต้องอยู่ระหว่าง 0-1")
        if width <= 0 or height <= 0:
            errors.append(f"line {line_number}: width/height ต้องมากกว่า 0")
    return errors

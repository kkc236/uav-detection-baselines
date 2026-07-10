from src.visdrone import convert_visdrone_row


def test_convert_visdrone_row_skips_ignored_region():
    assert convert_visdrone_row("10,20,30,40,0,4,0,0", image_width=100, image_height=200) is None


def test_convert_visdrone_row_to_yolo_box():
    row = convert_visdrone_row("10,20,30,40,1,4,0,0", image_width=100, image_height=200)

    assert row == "3 0.250000 0.200000 0.300000 0.200000"

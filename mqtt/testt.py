from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True) # need to run only once to load model into memory
img_path = 'gfg_dummy_pic.png'
result = ocr.ocr(img_path, det=False, rec=False, cls=True)
for line in result:
    print(line)

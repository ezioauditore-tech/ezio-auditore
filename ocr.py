from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = 'bike2.webp'
result = ocr.ocr(img_path, cls=True)
txts = [line[1][0] for line in result]
print(txts)
# print([line[0][1][0] for line in result])
# print([line[1][1][0] for line in result])
# print(result[0][0][1])
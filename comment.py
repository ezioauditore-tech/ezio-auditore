# limit the number of cpus used by high performance libraries
import os 
# checks a single component of code

import sys
sys.path.insert(0, './yolov5')
import argparse #Parser for command-line
import os
import platform #for retrieving info about the machine
import shutil  #for file manipulation(copy,paste,add,etc)
import time
from pathlib import Path #for file handling in os
import cv2 #computer vision
import torch #for creating deep neural networks
import torch.backends.cudnn as cudnn
from paddleocr import PaddleOCR,draw_ocr #ocr

#importing yolov5 and deepsort
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import timeit
import asyncio
import numpy as np
import imutils #for image processing purpose(translation,rotation,etc)
from PIL import Image #to deal with images



#OCR
async def function_asyc(crop_img):
    
    print(crop_img.shape)
    data = Image.fromarray(crop_img)
    data.save('gfg_dummy_pic.png')      
    img = cv2.imread('gfg_dummy_pic.png')
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    img = cv2.resize(img, (600,400) )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 
    edged = cv2.Canny(gray, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in contours:    
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                        
        if len(approx) == 4:
            screenCnt = approx
            break

        if screenCnt ==None:
            detected = 0
            print ("No contour detected")
        else:
            detected = 1

        if detected == 1 and False:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
            mask = np.zeros(gray.shape,np.uint8)
            new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
            new_image = cv2.bitwise_and(img,img,mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:np.int_(bottomx+1), topy:np.int_(bottomy+1)]
            result = ocr.ocr(Cropped, cls=True)
            Cropped = cv2.resize(Cropped,(400,200))
            print("programming_fever's License Plate Recognition\n")
            print("Detected license plate Number is:",result)
            img = cv2.resize(img,(500,300))
            Cropped = cv2.resize(Cropped,(400,200))
            cv2.imshow('car',img)
            cv2.imshow('Cropped',Cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('no number plate is detected')


FILE = Path(__file__).resolve() #shows the current path
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
data = []
temp_track = -1

#detecting the arguments that had been passed 
def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()


    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit) #loadstreams for loading the data
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size


    # Get names and colors of the classes:'car','bike',etc;
    names = model.module.names if hasattr(model, 'module') else model.names 
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        img = torch.from_numpy(img).to(device) #converting np array to tensors
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            start_time = timeit.default_timer()
            #Yolov5 detections
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                print('path is: ',p)
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                xywhs = xyxy2xywh(det[:, 0:4])#coordinates of x,y,w,h of detection 
                confs = det[:, 4] #accuracy
                clss = det[:, 5]
                print('whats this class :',clss)
                # pass detections to deepsort

                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                print(timeit.default_timer() - start_time)
                # x1, y1, x2, y2, track_id, class_id
                global temp_track
                print('this is output: ',outputs)
                print('this is s: ',s)
                if len(outputs) > 0:
                    px1, py1, px2, py2, t_id, cl_id = outputs[0]
                    if t_id != temp_track:
                        temp_track = t_id
                        np_img = im0.copy()
                        print(np_img.shape)
                        # cropped numpy array
                        crop_img = np_img[py1:py2, px1:px2]
                        asyncio.run(function_asyc(crop_img))
                        print('time:',timeit.default_timer() - start_time)
                        print(t_id)
                        
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        #count
                        count_obj(bboxes,w,h,id)
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        # print(label)
                        annotator.box_label(bboxes, label, color=colors(c, True))
                            
            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                global count
                color=(255,0,0)
                start_point = (0, h-50)
                end_point = (w, h-50)
                thickness = 3
                org = (150, 150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3
                cv2.putText(im0, str(count), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
def count_obj(box,w,h,id):
    global count,data
    if int(box[1]+(box[3]-box[1])/2) > (h -350):
        if  id not in data:
            count += 1
            data.append(id)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/Traffic.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
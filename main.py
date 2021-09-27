import sys
sys.path.insert(0, './yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import easyocr



yolo_weights = "crowdhuman_yolov5m.pt"
deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
source = 'SPart2.mp4'
out = 'inference/output'
img_size = 640
conf_thres = 0.4
iou_thres = 0.5
fourcc = 'mp4v'
save_vid = False
device = 0
classes = 0
config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"
imgsz = check_img_size(img_size)
show_vid = True
df = dict()
reader = easyocr.Reader(['en']) 


def updateDB(section,timeCal,DateCal,id):
    if id in df:
        if df[id][0] != section:
            with open('DB.csv','r+') as db:
                data = db.readlines()
                db.writelines(f'\nP{id},A{df[id][0]},{df[id][1]}  {df[id][2]},{df[id][1]} {df[id][3]}')
            df[id]  = [section,DateCal,timeCal,timeCal]
        else:
            df[id][3] = timeCal
                            
    else:
        df[id] =  [section,DateCal,timeCal,timeCal] # [section ,Date,entry, exit]     
    print(df)


def ocr(img,dy = 50,dx =155,flag="time"):
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    H,W = img.shape[:2]
    if(flag=="time"):
        reqFieldImg = img[H-dy:H,W-dx:W]
        textList = reader.readtext(reqFieldImg)
    else:
        reqFieldImg = img[H-dy:H,W-2*dx:W-dx+10]
        textList = reader.readtext(reqFieldImg)

    if(len(textList)>0):
        return textList[0][1]
    else:
        return None

def findSection(x,y,width,height,horizontal_section=4):
    x_section=(x+width-1)/width
    if y>=height: #we can write y>width as well
        x_section+=horizontal_section
    return int(x_section)

with torch.no_grad():
 # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    #attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    device = select_device(device)     
    
                       

    half = device.type != 'cpu'

    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

  
    if half:
        model.half()  # to FP16

    vid_path, vid_writer = None, None
    if show_vid:
        show_vid = check_imshow()
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'


    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

            # Inference
        t1 = time_sync()
        pred = model(img, augment=False)[0]

        pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes=classes, agnostic=False)
        t2 = time_sync()

        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                    # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                
                    # draw boxes for visualization
                if len(outputs) > 0:
                    timeCal = ocr(im0,flag="time")
                    DateCal = ocr(im0,flag="date")

                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                            
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        cv2.circle(im0,(bbox_left + bbox_w//2, bbox_top + bbox_h//2),4,(255,0,0),-1)
                        
                        H,W = im0.shape[:2]

                        section = findSection(bbox_left + bbox_w//2,bbox_top + bbox_h//2,W//4,H//2)
                        cv2.putText(im0,str(section),(bbox_left,bbox_top+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                        updateDB(section,timeCal,DateCal,id)
                       

            else:
                deepsort.increment_ages()

                # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
                
            im0 = annotator.result()
            H,W = im0.shape[:2]
            locations = {'A':(0,H//2),'B':(W,H//2),'C':(W//4,0),'D':(W//4,H),'E':(2*W//4,0),'F':(2*W//4,H),'G':(3*W//4,0),'H':(3*W//4,H),'I':(W//4,H//2),'J':(2*W//4,H//2),'K':(3*W//4,H//2)}
            #print(im0.shape)
            cv2.line(im0,locations['A'],locations['B'],(0,0,255),1)
            cv2.line(im0,locations['C'],locations['D'],(0,0,255),1)
            cv2.line(im0,locations['E'],locations['F'],(0,0,255),1)
            cv2.line(im0,locations['G'],locations['H'],(0,0,255),1)
            cv2.putText(im0,"1",(0,0+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(im0,"2",(locations['C'][0]+10,locations['C'][1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(im0,"3",(locations['E'][0]+10,locations['E'][1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(im0,"4",(locations['G'][0]+10,locations['G'][1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(im0,"5",(locations['A'][0],locations['A'][1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(im0,"6",(locations['I'][0]+10,locations['I'][1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(im0,"7",(locations['J'][0]+10,locations['J'][1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(im0,"8",(locations['K'][0]+10,locations['K'][1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            if show_vid:
                #im0 = cv2.resize(im0,None,fx=0.5,fy=0.5)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        
    if save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))



# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
import numpy as np
import math
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
data = []
id_tracker=[]
id_tracker2=[]
id_tracker3=[]
id_flag=[]
time_tracker={}
time_tracker2={}
av_speed={}
end_time={}
start_time={}
pass_time={}
av_time_1={}
av_time_2={}
av_session_1={}
av_session_2={}
violation_type=['Speeding', 'Red Light']


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

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

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

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # read each frame
   
        
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        # The amount of time used to process each frame is added to get the total processing time of the video
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
       

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            #w, h = im0.shape[1],im0.shape[0]
            # im0.shape[1]= 1280
            # im0.shape[0]= 720
            w, h = im0.shape[1],im0.shape[0]
            # dim = f'width:{w}, height:{h}'
            # print(dim)
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
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    # There are a lot of object
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        #count_obj(bboxes,w,h,id) #count
                        c = int(cls)  # integer class
                        #session_average(50, bboxes, w,h,id,110, im0,save_dir)
                        sa_v2(50, bboxes, w,h,id,85, im0,save_dir, vid_cap,frame_idx )
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True)) # Add the label
                        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
                

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')


            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')
        

            # Stream results
            im0 = annotator.result()
            if show_vid:
                global count
                color=(0,255,0) #green

                #The coordinate system starts from the top left, y increases as going down
                # upper_line = {"start_point":(700, h-900), "end_point":(1250, h-900)}
                # lower_line = {"start_point":(150, h-350), "end_point":(1700, h-350)}

                upper_section = [{"start_point":(750, h-875), "end_point":(1250, h-875)}, {"start_point":(700, h-850), "end_point":(1300, h-850)}]
                lower_section = [{"start_point":(310, h-300), "end_point":(1700, h-300)}, {"start_point":(200, h-150), "end_point":(1800, h-150)}]
                # start_point = (0, h-350)
                # end_point = (w, h-350)
                # Define the line location
                # cv2.line(im0, upper_line['start_point'], upper_line['end_point'], color, thickness=2)
                # cv2.line(im0, lower_line["start_point"], lower_line["end_point"], color, thickness=2)

                cv2.line(im0, upper_section[0]['start_point'], upper_section[0]['end_point'], (255,0,0), thickness=2)
                cv2.line(im0, upper_section[1]['start_point'], upper_section[1]['end_point'], color, thickness=2)

                cv2.line(im0, lower_section[0]['start_point'], lower_section[0]['end_point'], (255,0,0), thickness=2)
                cv2.line(im0, lower_section[1]['start_point'], lower_section[1]['end_point'], color, thickness=2)
                
                org = (150, 150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3
                thickness=3
                #cv2.putText(im0, str(count), org, font, fontScale, color, thickness, cv2.LINE_AA)

                # frames = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                #f = vid_cap.get(cv2.CAP_PROP_FPS)
                # print(frames, f, frame_idx)

                
                #Resize the window to 720p for laptop
                im0 = cv2.resize(im0.copy(), (1280,720))
                cv2.imshow(str(p), im0)
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
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

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    print("Speeding vehicle id: "+ f'{id_tracker3}')
    print("Start_time: "+ f'{start_time}')
    #print('av_1:'+ f'{av_session_1}')
    #print("time_tracker2: "+ f'{time_tracker2}')
    print("End_time: "+f'{end_time}')
    print("Passed time: "+f'{pass_time}')
    print("Average speed: "+ f'{av_speed}')
    print("Results saved to:" + f'{save_dir}')


    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #print(dt, seen)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

def count_obj(box,w,h,id):
    global count,data
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    if int(box[1]+(box[3]-box[1])/2) > (h -350):
        if  id not in data:
            count += 1
            data.append(id)

# def speed_estimation(estimated_distance, box, w, h, id, speed_threshold):
#     global id_tracker, pass_time, time_tracker, av_speed, end_time
#     vehicle_vertial_pos = int(box[1]+(box[3]-box[1])/2)
#     vehicle_horizontal_pos = int(box[1]+(box[3]-box[1])/2)
#     if (vehicle_vertial_pos > (h-900)) and (700 < vehicle_horizontal_pos < 1250):
#         if (id not in id_tracker) and vehicle_vertial_pos< (h-350):
#             id_tracker.append(id)
#             #record the start time
#             time_tracker[id]=time.time()
#         elif id in id_tracker: # The vehicle has passed the first line
#             if vehicle_vertial_pos >(h-350):
#                 #if id not in speeding_tracker:
#                 end_time[id] = time.time()
#                 pass_time[id] = end_time[id] -time_tracker[id]
#                 average_speed = int((estimated_distance/pass_time[id])*3.6)
#                 av_speed[id]= average_speed
#                 if average_speed < speed_threshold:
#                     id_tracker.remove(id)
#             else:
#                 pass
#         else:
#             pass
            
# def session_average(distance, box, w, h, id, speed_limit,im0,save_dir): # for each frame
#     global pass_time, start_time, end_time, time_tracker, time_tracker2, av_speed
#     global av_time_1, av_time_2, id_tracker, id_tracker2, id_tracker3
#     global av_session_1, av_session_2, violation_type
    
#     vehicle_vertial_pos = int(box[1]+(box[3]-box[1])/2)
#     vehicle_horizontal_pos = int(box[0] + (box[2]-box[0])/2)

#     if (vehicle_vertial_pos > (h-875)) and (vehicle_horizontal_pos < 1300):
#         if (id not in id_tracker) and vehicle_vertial_pos< (h-850):
#             id_tracker.append(id)
#             time_tracker[id]=time.time() # session 1 start time
#             start_time[id]= time_tracker[id]
#         elif id in id_tracker and vehicle_vertial_pos >= (h-850):
#             av_time_1[id] = time.time() # session 1 end time
#             av_session_1[id] = (av_time_1[id] + time_tracker[id])/2
#             id_tracker.remove(id)
#         elif id not in id_tracker and (vehicle_vertial_pos > (h-850)):
#             if (id not in id_tracker2) and vehicle_vertial_pos >(h-300): 
#                 time_tracker2[id]=time.time() # session 2 start time
#                 id_tracker2.append(id)
#             elif id in id_tracker2 and vehicle_vertial_pos >= (h-150):
#                 av_time_2[id]= time.time() # session 2 end time
#                 end_time[id] = av_time_2[id]
#                 av_session_2[id]= (av_time_2[id]+ time_tracker2[id])/2 
#                 pass_time[id] = av_session_2[id] - av_session_1[id]
#                 av_speed[id] = int((distance/pass_time[id])*3.6)
#                 id_tracker2.remove(id)
#                 if av_speed[id] > speed_limit and id not in id_tracker3:
#                     id_tracker3.append(id)
#     elif (vehicle_vertial_pos > (h-300)) and (vehicle_horizontal_pos < 1800): # Deal with exception (car too close to the camera)
#         if (id not in id_tracker2) and vehicle_vertial_pos >(h-300): 
#                 time_tracker2[id]=time.time()
#                 id_tracker2.append(id)
#         elif id in id_tracker2 and vehicle_vertial_pos >= (h-150):
#             av_time_2[id]= time.time()
#             end_time[id] = av_time_2[id]
#             av_session_2[id]= (av_time_2[id]+ time_tracker2[id])/2
#             pass_time[id] = av_time_2[id]-av_time_1[id]
#             av_speed[id] = int((distance/pass_time[id])*3.6)
#             id_tracker2.remove(id)
#             if av_speed[id] > speed_limit and id not in id_tracker3:
#                 id_tracker3.append(id)
#                 x,y,w,h=int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])                   
#                 img_ = im0.astype(np.uint8)
#                 crop_img=img_[y:y+ h, x:x + w]                          
#                 #!!rescale image !!!
#                 filename= f'{id}'+ f'{violation_type[0]}'+ '.jpg' # Have to add the time information as well
#                 filepath=os.path.join(save_dir, filename)
#                 print(filepath)
#                 cv2.imwrite(filepath, crop_img)

def sa_v2(distance, box, w, h, id, speed_limit,im0,save_dir, vid_cap, frame_index): # for each frame
    # distance: the actutal estimated distance in meter
    # box: the coordinates of the bounding box as an array
    # w, h : width and height of video (frame)
    # id: car id as an integer
    # speed limit: user defined speed threshold
    # im0: the current frame image
    # save_dir: the directory to be saved to
    # vid_cap: cv2.VideoCapture(path)
    # frame_index: the current frame index (current frame)
    global id_tracker, pass_time, time_tracker,time_tracker2, av_speed, end_time, av_time_1, av_time_2, id_tracker2, id_tracker3, start_time, av_session_1, av_session_2,id_flag, violation_type
    
    vehicle_vertial_pos = int(box[1]+(box[3]-box[1])/2)
    vehicle_horizontal_pos = int(box[0] + (box[2]-box[0])/2)

    if (vehicle_vertial_pos > (h-875)) and (vehicle_horizontal_pos < w) and id not in id_tracker2:
        if (id not in id_tracker) and vehicle_vertial_pos< (h-850):
            id_tracker.append(id)
            time_tracker[id]=time.time() # session 1 start time
            start_time[id]= time_tracker[id]
            #print(id_tracker)
        elif id in id_tracker and vehicle_vertial_pos >= (h-850):
            av_time_1[id] = time.time() # session 1 end time
            av_session_1[id] = (av_time_1[id] + time_tracker[id])/2
            id_tracker.remove(id)
            #print("pass:"+ f'{id_tracker}')
        elif id not in id_tracker and vehicle_vertial_pos >=(h-300):
            time_tracker2[id]=time.time() # session 2 start time
            id_tracker2.append(id)
            id_flag.append(id)
            #print("pass2:"+ f'{id_tracker2}')

    elif id in id_tracker2 and (vehicle_horizontal_pos < w): # it passed the session 2 start line
        if vehicle_vertial_pos >= (h-150) and id in id_flag: # just need to record once so added the flag
            av_time_2[id]= time.time() # session 2 end time
            end_time[id] = av_time_2[id]
            av_session_2[id]= (av_time_2[id]+ time_tracker2[id])/2 
            pass_time[id] = av_session_2[id] - av_session_1[id]
            av_speed[id] = int((distance/pass_time[id])*3.6)
            #id_tracker2.remove(id)
            id_flag.remove(id)
            if av_speed[id] > speed_limit and id not in id_tracker3:
                id_tracker3.append(id)
                x,y,w,h=int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])                   
                img_ = im0.astype(np.uint8)
                crop_img=img_[y:y+ h, x:x + w]                          
                #!!rescale image !!!
                # Get the fps (frame per second)
                f = vid_cap.get(cv2.CAP_PROP_FPS)
                # Calculate the time the violation occurs
                time_in_sec = round(frame_index/f, 2)
                # Save the violated car as a pic and name it
                

                filename= f'{id}'+ '_'+ f'{violation_type[0]}'+ '_'+f'{time_in_sec}'+'.jpg' # Have to add the time information as well
                print('Vehicle ID: '+ f'{id} '+ 'Violation type: '+ f'{violation_type[0]} '+ 'Time occured: '+ f'{time_in_sec}\n')       
                # Save it to a user defined path
                filepath=os.path.join(save_dir, filename)
                completeName = os.path.join(save_dir, 'output.txt') 
                #print(filepath)
                # Get the cropped pic and save to path
                file1 = open(completeName, "a")
                toFile = 'Vehicle ID: '+ f'{id} '+ 'Violation type: '+ f'{violation_type[0]} '+ 'Time occured: '+ f'{time_in_sec}'
                file1.write(toFile+'\n')
                file1.close()
                cv2.imwrite(filepath, crop_img)
        else:
            pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
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

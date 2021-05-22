import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized



import streamlit as st
# import tensorflow as tf
# import numpy as np
from PIL import Image
# import pydot
# import graphviz
# import pydotplus

from streamlit.delta_generator import DeltaGenerator
# from tensorflow.keras.models import *
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
# import os
# import cv2
from PIL import Image




def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    # if device.type != 'cpu':
        # model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    print(save_path)
                    cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return save_path


# def global_upload_file(uploaded_file):
#         global image_file 
#         image_file= uploaded_file

if __name__ == '__main__':
    st.title("""
      Detection and localization abnormalities in chest X-Ray image
    """)
    st.write("\n")

    col_1, col_2= st.beta_columns(2)

    # with col_1:
    col_1.header("Upload Image")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    img_file = st.file_uploader(" ",type=['png', 'jpg', 'jpeg'],key=0)
    # accept_multiple_files=True
    # print(img_file.read())
    if img_file:
        image_to_predict = "/home/huynhmanhhung441/BK/LuanVanTotNghiep/Demo_detect/Dataset/test/"+img_file.name
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='/home/huynhmanhhung441/BK/LuanVanTotNghiep/Demo_detect/yolov5/runs/train/weights/best.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default= image_to_predict , help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.009, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', default= True, action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)
        check_requirements(exclude=('pycocotools', 'thop'))
        result_path=''
        with st.spinner('Classifying ...'):
            with torch.no_grad():
                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['/home/huynhmanhhung441/BK/LuanVanTotNghiep/Demo_detect/yolov5/runs/train/weights/best.pt']:
                        result_path = detect()
                        strip_optimizer(opt.weights)
                else:
                    result_path = detect()
        image = Image.open(result_path)
        st.image(image, caption='predicted image')

    # col_2.header('Prediction result')
    # if col_2.button("Click Here to Classify"):
    
        # col_2.header("Algorithm Predicts: ")
    #     st.write("Your chance of getting melanoma is", '\n')
        # st.write('**Probability: ** {:.2f} %'.format(prediction_number*100))


    

        # def load_trained_model():
    #    model_1 = load_model (PATH of your trained models)
    #    return model_1

    # model_1= load_trained_model()

    

    # def preprocess_image (image):
    #   image = np.expand_dims(image, axis=0)
    #   image = tf.image.resize(image, size=(256,256))
    #   image = tf.cast(image, dtype='float32')/255.0
    #   return image

    # def predict(image):
    #   result_1= model_1.predict(image)
    #   return result_1


    # import os
    # os.listdir("")
    
    # os.listdir("")


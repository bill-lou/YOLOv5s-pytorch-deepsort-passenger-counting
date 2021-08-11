import sys

sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized

# deepsort v1
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# deepsort v2
from deep_sort_alt.deepsort import deepsort_rbc

import argparse
import os
import platform
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)

mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

# blue polygon line
list_pts_blue = [[0, 994], [1920, 653], [1920, 578], [0, 951]]
ndarray_pts_blue = np.array(list_pts_blue, np.int32)
polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue],
                                    color=1)
polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

# yellow polygon
mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
list_pts_yellow = [[0, 1031], [1920, 734], [1920, 654], [0, 994]]
ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow],
                                      color=2)
polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2
polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow,
                                          (960, 540))

blue_color_plate = [255, 0, 0]
blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

yellow_color_plate = [0, 255, 255]
yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

color_polygons_image = blue_image + yellow_image

# list 与蓝色polygon重叠
list_overlapping_blue_polygon = []

# list 与黄色polygon重叠
list_overlapping_yellow_polygon = []

down_count = 0
up_count = 0

font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
draw_text_postion = (int(960 * 0.01), int(540 * 0.05))


def check_if_rect_overlaps_line(boundary_box, track_id):
    x1, y1, x2, y2 = [int(i) for i in boundary_box]
    y1_offset = int(y1 + ((y2 - y1) * 0.6))

    y = y1_offset
    x = x1

    if polygon_mask_blue_and_yellow[y, x] == 1:
        if track_id not in list_overlapping_blue_polygon:
            list_overlapping_blue_polygon.append(track_id)
            pass

        if track_id in list_overlapping_yellow_polygon:
            list_overlapping_yellow_polygon.remove(track_id)

            # up_count += 1
            return True, False
        else:
            pass

    elif polygon_mask_blue_and_yellow[y, x] == 2:
        if track_id not in list_overlapping_yellow_polygon:
            list_overlapping_yellow_polygon.append(track_id)
            pass

        if track_id in list_overlapping_blue_polygon:
            list_overlapping_blue_polygon.remove(track_id)

            # down_count += 1
            return False, True
        else:
            pass

    else:
        pass

    return False, False


def cleanup_tracking_lists(id_list):
    list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon

    for id1 in list_overlapping_all:
        is_found = False
        for bbox_id in id_list:
            if bbox_id == id1:
                is_found = True
                break
            pass
        pass

        if not is_found:
            if id1 in list_overlapping_yellow_polygon:
                list_overlapping_yellow_polygon.remove(id1)
            pass
            if id1 in list_overlapping_blue_polygon:
                list_overlapping_blue_polygon.remove(id1)
            pass
        pass

    list_overlapping_all.clear()


def add_polygons_image(frame):
    global color_polygons_image
    frame = cv2.add(frame, color_polygons_image)
    return frame


def draw_up_down_text(frame, up_count, down_count):
    text_draw = 'DOWN:' + str(down_count) + \
                ' UP:' + str(up_count)

    frame = cv2.putText(img=frame,
                        text=text_draw,
                        org=draw_text_postion,
                        fontFace=font_draw_number,
                        fontScale=1,
                        color=(45, 167, 45),
                        thickness=2)

    return frame


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def bbox_rel2(*xyxy):
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())

    return xyxy[0], xyxy[1], bbox_w, bbox_h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        draw_box(img, box, id, offset)


def draw_box(img, boundary_box, id, offset=(0, 0)):
    x1, y1, x2, y2 = [int(i) for i in boundary_box]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]

    color = compute_color_for_labels(id)
    label = '{}{:d}'.format("", id)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4),
                  color, -1)
    cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,
                2, [255, 255, 255], 2)


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize

    # create folder if no exists
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model

    # load to FP32
    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string

            filename = Path(p).name
            save_path = str(Path(out) / filename)

            fileSp = os.path.splitext(filename)

            if len(fileSp) >= 2:
                basename = fileSp[0]
                extension = fileSp[1]
                save_path = str(Path(out) / Path(basename + "_v1" + extension))

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # xywhs <class 'torch.Tensor'>
                # confss <class 'numpy.ndarray'>
                # im0 <class 'numpy.ndarray'>

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                    # check if boxes overlap lines
                    for i, box in enumerate(bbox_xyxy):
                        id = int(
                            identities[i]) if identities is not None else 0
                        up, down = check_if_rect_overlaps_line(box, id)

                        if up:
                            global up_count
                            up_count += 1

                        if down:
                            global down_count
                            down_count += 1

                    cleanup_tracking_lists(identities)
                else:
                    list_overlapping_blue_polygon.clear()
                    list_overlapping_yellow_polygon.clear()

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') %
                                    (frame_idx, identity, bbox_left, bbox_top,
                                     bbox_w, bbox_h, -1, -1, -1,
                                     -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # resize polygon image 1080 -> ?
            width = im0.shape[1]
            height = im0.shape[0]

            global color_polygons_image
            color_polygons_image = cv2.resize(color_polygons_image,
                                              (width, height))
            global polygon_mask_blue_and_yellow
            polygon_mask_blue_and_yellow = cv2.resize(
                polygon_mask_blue_and_yellow, (width, height))

            im0 = draw_up_down_text(im0, up_count, down_count)
            im0 = add_polygons_image(im0)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release(
                            )  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc),
                            fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


def detectAlternative(opt):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    deepsort = deepsort_rbc(opt.deepsort_wheight)

    # Initialize

    # create folder if no exists
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model

    # load to FP32
    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string

            filename = Path(p).name
            save_path = str(Path(out) / filename)

            fileSp = os.path.splitext(filename)

            if len(fileSp) >= 2:
                basename = fileSp[0]
                extension = fileSp[1]
                save_path = str(Path(out) / Path(basename + "_v2" + extension))

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel2(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # xywhs <class 'torch.Tensor'>
                # confss <class 'numpy.ndarray'>
                # im0 <class 'numpy.ndarray'>

                # Pass detections to deepsort
                tracker, detections_class = deepsort.run_deep_sort(
                    im0, confss, xywhs)

                id_list = list()

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    #Get the corrected/predicted bounding box
                    bbox = track.to_tlbr()

                    # features = track.features  #Get the feature vector corresponding to the detection.

                    draw_box(im0, bbox, track.track_id)

                    up, down = check_if_rect_overlaps_line(
                        bbox, track.track_id)

                    id_list.append(track.track_id)

                    if up:
                        global up_count
                        up_count += 1

                    if down:
                        global down_count
                        down_count += 1

                    # Write MOT compliant results to file
                    if save_txt:
                        bbox_left = bbox[0]
                        bbox_top = bbox[1]
                        bbox_w = bbox[2]
                        bbox_h = bbox[3]
                        identity = track.track_id
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') %
                                    (frame_idx, identity, bbox_left, bbox_top,
                                     bbox_w, bbox_h, -1, -1, -1,
                                     -1))  # label format

                if len(id_list) > 0:
                    cleanup_tracking_lists(id_list)
                else:
                    list_overlapping_blue_polygon.clear()
                    list_overlapping_yellow_polygon.clear()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # resize polygon image 1080 -> ?
            width = im0.shape[1]
            height = im0.shape[0]

            global color_polygons_image
            color_polygons_image = cv2.resize(color_polygons_image,
                                              (width, height))
            global polygon_mask_blue_and_yellow
            polygon_mask_blue_and_yellow = cv2.resize(
                polygon_mask_blue_and_yellow, (width, height))

            im0 = draw_up_down_text(im0, up_count, down_count)
            im0 = add_polygons_image(im0)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path

                        # release previous video writer
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc),
                            fps, (w, h))

                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='yolov5/weights/yolov5x.pt',
                        help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='in', help='source')
    parser.add_argument('--output',
                        type=str,
                        default='out',
                        help='output folder')  # output folder
    parser.add_argument('--img-size',
                        type=int,
                        default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.4,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.5,
                        help='IOU threshold for NMS')
    parser.add_argument('--fourcc',
                        type=str,
                        default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        default=[0],
                        help='filter by class')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort",
                        type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--deepsort_wheight",
                        type=str,
                        default="ckpts/model80.pt")
    parser.add_argument('--deepsort',
                        type=int,
                        default=1,
                        help='which deepsort version 1 or 2')

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        if args.deepsort == 1:
            print("detect with deepsort v1")
            detect(args)
        elif args.deepsort == 2:
            print("detect with deepsort v2")
            detectAlternative(args)
        else:
            print("deepsort version not found")

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import os.path as osp
from glob import glob
import shutil
from tqdm import tqdm

classes = [ "person","head","door_open","door_half_open","door_close"]

abs_path = os.getcwd()
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xmlfile, labelpath):
    xmlname = xmlfile.split('/')[-1].split('.')[0]
        
    out_file = open(join(labelpath, xmlname + '.txt'), 'w')
    tree=ET.parse(xmlfile)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        #difficult = obj.find('difficult').text
        name = obj.find('name').text
        
        cls_id = classes.index(name)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        # s = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
        # print(s)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                
    out_file.close()


if __name__ == '__main__': 
    projectbase = '/project/train/dataset'
    labelpath = join(projectbase, 'labels')
    saveimg = join(projectbase, 'images')

    if not osp.exists(labelpath):
        os.mkdir(labelpath)

    if not osp.exists(saveimg):
        os.mkdir(saveimg)
    
    for img in tqdm(glob(projectbase + "/*/*.jpg")):
            

        if 'xml' in img:
            continue
        imgname = img.split('/')[-1]
        dstimg = join(saveimg, imgname)
        shutil.copy(img, dstimg)

        xmlfile = img.split('.')[0] + '.xml'

        convert_annotation(xmlfile, labelpath)
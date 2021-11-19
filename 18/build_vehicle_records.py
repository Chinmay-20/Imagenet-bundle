from config import dlib_front_rear_config as config
from tfannotation import TFAnnotation
from bs4 import BeautifulSoup
from PIL import Image
import tensorflow as tf
import os

def main(_):
	f=open(config.CLASSES_FILE, "w")
	
	for (k,v) in config.CLASSES.items():
		item=("item {\n"
			"\tid: "+ str(v) + "\n"
			"\tname: '"+k+"'\n"
			"}\n")
		f.write(item)
		
	f.close()
	
	datasets=[
		("train",config.TRAIN_XML,config.TRAIN_RECORD),
		("test",config.TEST_XML,config.TEST_RECORD)
	]
	
	for(dType,inputPath,outputPath) in datasets:
		print("[INFO] processing '{}'...".format(dType))
		contents=open(inputPath).read()
		soup=BeautifulSoup(contents,"html.parser")
		
		writer=tf.python_io.TFRecordWriter(outputPath)
		
		total=0
		
		for image in soup.find_all("image"):
			p=os.path.sep.join([config.BASE_PATH,image["file"]])
			encoded=tf.gfile.GFile(p,"rb").read()
			encoded=bytes(encoded)
			
			pilImage=Image.open(p)
			(w,h)=pilImage.size[:2]
			
			filename=image["file"].split(os.path.sep)[-1]
			encoding=filename[filename.rfind(".")+1:]
			
			tfAnnot=TFAnnotation()
			tfAnnot.image=encoded
			tfAnnot.encoding=encoding
			tfAnnot.filename=filename
			tfAnnot.width=w
			tfAnnot.height=h
			
			
			for box in image.find_all("box"):
				if box.has_attr("ignore"):
					continue
				
				startX=max(0,float(box["left"]))
				startY=max(0,float(box["top"]))
				endX=min(w,float(box["width"])+startX)
				endY=min(h,float(box["height"])+startY)
				label=box.find("label").text
				
				
				xMin=startX/w
				xMax=endX/w
				yMin=startY/h
				yyMax=endY/h
				
				if xMin>xMax or yMin>yMax:
					continue
					
				elif xMax<xMin or yMax<yMin:
					continue
					
				tfAnnot.xMins.append(xMin)
				tfAnnot.xMaxs.append(xMax)
				tfAnnot.yMins.append(yMin)
				tfAnnot,yMaxs.append(yMax)
				tfAnnot.textLabels.append(label.encode("utf8"))
				tfAnnot.classes.append(config.CLASSES[label])
				tfAnnot.difficult.append(0)
				
				total+=1
			features=tf.train.Features(feature=tfAnnot.build())
			example=tf.train.Example(features=features)
			
			wrtiter.write(example.SerializeToString())
			
			writer.close()
			
			print("[INFO] {} examples saved for '{}'".format(total,dType))
			
if __name__=="__main__":
	tf.app.run()

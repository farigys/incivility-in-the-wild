import datetime
import glob
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from textblob import TextBlob as tb
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import operator
import random
from sklearn.model_selection import train_test_split
import math

def tf(word, blob):
	return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
	return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
	try:
		return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
	except:
		return 0

def tfidf(word, blob, bloblist):
	return tf(word, blob) * idf(word, bloblist)

def replace_article_name(title):
	if title == "BUSINESSREDUCING": title = "CITYREDUCING"
	elif title == "CHUYS": title = "3CHUYS" 
	elif title == "JUSTGOWITH": title = "JUSTGO"
	elif title == "IRAQOVER": title = "IRAQWAR" 
	elif title == "NEWSALESTAX": title = "NEWSALES"
	elif title == "GARDENTAKES": title = "GARDENTAKERS"
	elif title == "ANILL": title = "ANILLADVISED"
	elif title == "SOUTHSIDETEXAS": title = "SOUTHSIDE"
	elif title == "STOPANDFRISK": title = "STOPAND"
	elif title == "GOVPERRYS": title = "GOVPERRY"
	elif title == "REASONSTO": title = "REASONTO"
	elif title == "E!SAYS": title = "ESAYS"
	elif title == "HOMEBAKED": title = "HOMEBAKEDCOOKIES"
	elif title == "AFOODWORLD": title = "AFOOD"
	elif title == "AEUROPESOUTHWEST": title = "AEUROPE"
	elif title == "ADOPTION": title = "ADOPTIONCREATES"
	elif title == "THERESSTILL": title = "THERESTILL"
	elif title == "PAC12CROSS": title = "PAC12"
	elif title == "MASKEDMEN": title = "MASKEDMAN"
	elif title == "POLICESHOOTINGS": title = "POLICESHOOTING"
	elif title == "OCT22TODAY": title = "OCT22"
	return title

incivility_types = ["NAMECALLING","ASPERSION","LYING","VULGARITY","PEJORATIVE","SARCASM","OTHER INCIVILITY","HYPERBOLE","NONCOOPERATION","OFFTOPIC"]

#incivility_types = ["namecalling", "aspersion", "lying", "pejorative", "sarcasm", "hyperbole", "noncoop", "offtopic", "others", "vulgarity"]
#incivility_types = ['aspersion', 'vulgarity', 'lying', 'namecalling', 'pejorative'] #used for sanity checking

problem_keys = ["20111103/Opinion/OPINIONCAINSFAULTY.pdf","20111028/State News/STATENEWSPEARCEHOLDS.pdf","20111102/State News/STATENEWSARIZONAREDISTRICTING.pdf","20111106/Business/BUSINESSRAISINGHAY.pdf","20111023/Local News/LOCALNEWS2HUNTERS.pdf","20111104/Business/BUSINESSSCICHANNEL.pdf"]

def folder_names(): #deprecated
	root = "../data/complete_data/"

	start = datetime.datetime.strptime("17-10-2011", "%d-%m-%Y")
	end = datetime.datetime.strptime("14-11-2011", "%d-%m-%Y")
	date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

	topics = set()
	dates = []

	for date in date_generated:
		dt = date.strftime("%Y%m%d")
		dates.append(dt)
		listofdirs = glob.glob(root + dt + "/*/")
		for dir in listofdirs: topics.add(dir.split("/")[-2])

	return dates, topics

def read_excel():
	file = '../ADS_Civility_Data_Trimmed.xls'
	xl = pd.ExcelFile(file)
	df1 = xl.parse('Sheet1')
	columns = df1.columns.values
	rowcount = df1.shape[0]
	topic_set = set() 
	for t in df1.ix[0:rowcount,"SECTION"]:
		if type(t) == float: continue
		topic_set.add(t.strip())
	return df1, topic_set

def replace_chars(text):
	to_replace = " '-,./?:(){}[]"
	for c in to_replace:
	        text = text.replace(c, "")
	return text

def extract_to_check(): #deprecated
	#incivility_types = ["namecalling", "aspersion", "lying", "pejorative", "sarcasm", "hyperbole", "noncoop", "offtopic", "others", 'vulgarity']
	#print folder_names()[1]
	df, topic_set = read_excel()
	columns = df.columns.values
	rowcount = df.shape[0]
	incivility_dict = dict()

	#print topic_set

	incivility_dict = dict()

	foldername_set = set()

	comment_dict = dict()

	for i in range(rowcount):
		try:
			title = replace_chars(df.ix[i,"ARTICLE_TITLE"].upper()).strip()
			title = title.replace("PROTESTOR","PROTESTER")
			
			title = replace_article_name(title)
			 
			foldername = df.ix[i,"SECTION"].strip()
			date = str(df.ix[i,"DATE_OF_ARTICLE"]).strip()

			comment_num = str(int(df.ix[i,"COMMENT NUMBER"]))

			if "Nation" in foldername: foldername = "Nation and World"
			if "Lifestyle" in foldername: foldername = "Lifestyles"
			if "Local" in foldername: foldername = "Local News"
			if "State" in foldername: foldername = "State News"
			
			foldername_set.add(foldername)
			
			section = foldername.replace(" ","").upper()

			updated_date = ""

			#if "-" in date:
			#	parts = date.split("-")
			#	updated_date = parts[0] + parts[1] + parts[2]

			if len(date) == 6:
				updated_date = "2011" + date[0:2] + date[2:4]
			elif len(date) == 5:
				updated_date =  "2011" + date[0:2] + "0" + date[2]
			#else: print date

			key = updated_date + "/" + foldername + "/" + section + title + ".pdf"

			if key in comment_dict:
				comment_dict[key]+=1
			else: comment_dict[key] = 1
		except:
			continue

	sorted_dict = sorted(comment_dict.items(), key=operator.itemgetter(1), reverse=True)
	
	ext_com_dict = dict()

	lines = open("../comment-stat", "r").readlines()

	for line in lines:
		parts = line.strip().split(":")
		key = parts[0]
		count = int(parts[1])
		if count == 0: continue
		if key in ext_com_dict: ext_com_dict[key] += count
		else: ext_com_dict[key] = count
	
	#try:
	count = 0	
	for e in sorted_dict:
		if e[0] == "20111018/Entertainment/ENTERTAINMENTDEARABBY.pdf": continue
		if e[1] != ext_com_dict[e[0]]: print e[0], e[1], ext_com_dict[e[0]]
		else: count += int(e[1])
	#except Exception as ex:
	#	print ex, comment_dict[key]
	print count	
			


def extract_and_categorize(): #deprecated
	#incivility_types = ["namecalling", "aspersion", "lying", "pejorative", "sarcasm", "hyperbole", "noncoop", "offtopic", "others", "vulgarity"]
	#print folder_names()[1]
	df, topic_set = read_excel()
	columns = df.columns.values
	rowcount = df.shape[0]
	incivility_dict = dict()

	#print topic_set

	incivility_dict = dict()

	foldername_set = set()

	for i in range(rowcount):
		incivility_counts = []
		try:
			title = replace_chars(df.ix[i,"ARTICLE_TITLE"].upper()).strip()
			title = title.replace("PROTESTOR","PROTESTER")
			title = replace_article_name(title)
 
			foldername = df.ix[i,"SECTION"].strip()
			date = str(df.ix[i,"DATE_OF_ARTICLE"]).strip()

			comment_num = str(int(df.ix[i,"COMMENT NUMBER"]))

			for inc_type in incivility_types:
				incivility_counts.append(str(df.ix[i,inc_type]).strip())

 
			if "Nation" in foldername: foldername = "Nation and World"
			if "Lifestyle" in foldername: foldername = "Lifestyles"
			if "Local" in foldername: foldername = "Local News"
			if "State" in foldername: foldername = "State News"
			
			foldername_set.add(foldername)
			
			section = foldername.replace(" ","").upper()

			updated_date = ""

			#if "-" in date:
			#	parts = date.split("-")
			#	updated_date = parts[0] + parts[1] + parts[2]

			if len(date) == 6:
				updated_date = "2011" + date[0:2] + date[2:4]
			elif len(date) == 5:
				updated_date =  "2011" + date[0:2] + "0" + date[2]
			#else: print date

			key = updated_date + "/" + foldername + "/" + section + title + "_" + comment_num

			incivility_dict[key] = incivility_counts

			#print title, section, foldername, updated_date, comment_num, namecalling, aspersion, lying, pejorative, sarcasm, hyperbole, noncoop, offtopic, offtopic, others
			


		except Exception as e:
			print e			
			continue
			#print df.ix[i,"ARTICLE_TITLE"], df.ix[i,"SECTION"], df.ix[i,"Date"]
	#for k in incivility_dict.keys():
	#	if "20111019/Lifestyles/" in k: print k, incivility_dict[k]
	
	comment_dict = dict()

	print len(incivility_dict)

	root = "../data/complete_data/"
	lines = open("../comment-stat", "r").readlines()

	fw_list = []

	fw1 = open("none.txt", "w")
	
	for f in incivility_types:
		fw_list.append(open(f + ".txt", "w"))
	
	for line in lines:
		part_1 = line.strip().split(":")
		comment_count = part_1[1]
		if part_1[0] in problem_keys: continue
		parts = part_1[0].split("/")
		date = parts[0]
		section = parts[1]
		title = parts[2].split(".")[0]
		#print date, section, title
		#print part_1[0]
		ls = open(root + part_1[0] + ".txt", "r").readlines()
		comment_id = 1		
		comment = ""
		for l in ls:
			if "&&&&&&&&&&&&&&&&&&&&" in l:
				#print comment_id
				#print comment.strip()
				#print "&&&&&&&&&&&&&&&&&&&&"
				#temp = incivility_dict[part_1[0].split(".")[0] + "_" + str(comment_id)]
				try: incivility = incivility_dict[part_1[0].split(".")[0] + "_" + str(comment_id)]
				except: 
					print part_1[0].split(".")[0] + "_" + str(comment_id)
					comment = ""				
					comment_id+=1
					continue
				yes_count = 0					
				for inc in range(len(incivility)):
					#print incivility[inc] 
					if incivility[inc].lower() == "yes":
						yes_count += 1 
						fw_list[inc].write(comment.strip() + "\n")
				if yes_count == 0: 
					fw1.write(comment.strip() + "\n")
				comment = ""				
				comment_id+=1
								
			else:
				#print "dhuksi"
				comment+=l

	fw1.close()
	for f in fw_list:
		f.close()

	#print folder_names()[1]
	#print foldername_set

def extract_example(): #sanity check
	#print folder_names()[1]
	df, topic_set = read_excel()
	columns = df.columns.values
	rowcount = df.shape[0]
	incivility_dict = dict()

	#print topic_set

	incivility_dict = dict()

	foldername_set = set()

	for i in range(rowcount):
		incivility_counts = []
		try:
			title = replace_chars(df.ix[i,"ARTICLE_TITLE"].upper()).strip()
			title = title.replace("PROTESTOR","PROTESTER")
			title = replace_article_name(title)
 
			foldername = df.ix[i,"SECTION"].strip()
			date = str(df.ix[i,"DATE_OF_ARTICLE"]).strip()

			comment_num = str(int(df.ix[i,"COMMENT NUMBER"]))


			for inc_type in incivility_types:
				incivility_counts.append(str(df.ix[i,inc_type]).strip())
			
			if "Nation" in foldername: foldername = "Nation and World"
			if "Lifestyle" in foldername: foldername = "Lifestyles"
			if "Local" in foldername: foldername = "Local News"
			if "State" in foldername: foldername = "State News"
			
			foldername_set.add(foldername)
			
			section = foldername.replace(" ","").upper()

			updated_date = ""

			#if "-" in date:
			#	parts = date.split("-")
			#	updated_date = parts[0] + parts[1] + parts[2]

			if len(date) == 6:
				updated_date = "2011" + date[0:2] + date[2:4]
			elif len(date) == 5:
				updated_date =  "2011" + date[0:2] + "0" + date[2]
			#else: print date

			key = updated_date + "/" + foldername + "/" + section + title + "_" + comment_num

			incivility_dict[key] = incivility_counts

			#print title, section, foldername, updated_date, comment_num, namecalling, aspersion, lying, pejorative, sarcasm, hyperbole, noncoop, offtopic, offtopic, others
			


		except Exception as e:
			print e			
			continue
			#print df.ix[i,"ARTICLE_TITLE"], df.ix[i,"SECTION"], df.ix[i,"Date"]
	#for k in incivility_dict.keys():
	#	if "20111019/Lifestyles/" in k: print k, incivility_dict[k]
	
	comment_dict = dict()

	print len(incivility_dict)

	root = "../data/complete_data/"
	lines = open("../comment-stat", "r").readlines()

	fw1 = open("../sanity_check_1.txt", "w")
	
	sidx = 1

	total_comment = 0
	
	for line in lines:
		if total_comment == 500:
			fw1.close() 
			break
		part_1 = line.strip().split(":")
		comment_count = part_1[1]
		if part_1[0] in problem_keys: continue
		parts = part_1[0].split("/")
		date = parts[0]
		section = parts[1]
		title = parts[2].split(".")[0]
		#print date, section, title
		#print part_1[0]
		ls = open(root + part_1[0] + ".txt", "r").readlines()
		comment_id = 1		
		comment = ""
		for l in ls:
			if "&&&&&&&&&&&&&&&&&&&&" in l:
				#print comment_id
				#print comment.strip()
				#print "&&&&&&&&&&&&&&&&&&&&"
				#temp = incivility_dict[part_1[0].split(".")[0] + "_" + str(comment_id)]
				try: incivility = incivility_dict[part_1[0].split(".")[0] + "_" + str(comment_id)]
				except: 
					#print part_1[0].split(".")[0] + "_" + str(comment_id)
					comment = ""				
					comment_id+=1
					continue

				rand = random.randint(1,101)
				if rand%10 == 0 and "quote" not in comment.lower() and "wrote" not in comment.lower() and "[This post has been removed]" not in comment:
					total_comment += 1
					if total_comment == 500: break
					if total_comment%100 == 0:
						fw1.close()
						sidx+=1
						fw1 = open("sanity_check_" + str(sidx) + ".txt", "w")
					fw1.write(comment.strip() + "\n")
					fw1.write("-----------------------------------------------------------\n")					
					for i, inc in enumerate(incivility):
						fw1.write(incivility_types[i] + ": " + inc + "\n")
					fw1.write("-----------------------------------------------------------\n")
					fw1.write("-----------------------------------------------------------\n") 
					
				comment = ""				
				comment_id+=1
								
			else:
				#print "dhuksi"
				comment+=l

	fw1.close()

	#print folder_names()[1]
	#print foldername_set

def create_metadata_file(): #deprecated
	#print folder_names()[1]
	df, topic_set = read_excel()
	columns = df.columns.values
	rowcount = df.shape[0]

	print df.shape

	incivility_dict = dict()

	#print topic_set

	incivility_dict = dict()

	foldername_set = set()

	data = dict()
	
	data["id"] = []
	data["Name"] = []
	data["ThumbsUp"] = []
	data["ThumbsDown"] = []
	data["Section"] = []

	usermap = set()



	for i in range(rowcount):
		try:
			usermap.add(str(df.ix[i,"Name"]).strip().encode("utf-8"))
		except:
			continue
	
	userlist = list(usermap)

	fwuser = open("../data/user_list.csv", "w")

	for ui in range(len(userlist)):
		fwuser.write(userlist[ui] + "," + str(ui) + "\n")
	
	fwuser.close()
	
	for i in range(rowcount):
		incivility_counts = []
		try:
			title = replace_chars(df.ix[i,"ARTICLE_TITLE"].upper()).strip()
			title = title.replace("PROTESTOR","PROTESTER")
			title = replace_article_name(title)
 
			foldername = df.ix[i,"SECTION"].strip()
			date = str(df.ix[i,"DATE_OF_ARTICLE"]).strip()

			comment_num = str(int(df.ix[i,"COMMENT NUMBER"]))

			data["Name"].append(userlist.index(str(df.ix[i,"Name"]).strip().encode("utf-8")))
			data["ThumbsUp"].append(str(df.ix[i,"ThumbsUp"]).strip().encode("utf-8"))
			data["ThumbsDown"].append(str(df.ix[i,"ThumbsDown"]).strip().encode("utf-8"))
			#data["Journalist"].append(str(df.ix[i,"JOURNALIST'S NAME"]).strip().encode("utf-8"))

 
			if "Nation" in foldername: foldername = "Nation and World"
			if "Lifestyle" in foldername: foldername = "Lifestyles"
			if "Local" in foldername: foldername = "Local News"
			if "State" in foldername: foldername = "State News"
			
			foldername_set.add(foldername)
			
			section = foldername.replace(" ","").upper()

			updated_date = ""

			#if "-" in date:
			#	parts = date.split("-")
			#	updated_date = parts[0] + parts[1] + parts[2]

			if len(date) == 6:
				updated_date = "2011" + date[0:2] + date[2:4]
			elif len(date) == 5:
				updated_date =  "2011" + date[0:2] + "0" + date[2]
			#else: print date

			key = updated_date + "/" + foldername + "/" + section + title + "_" + comment_num

			data["Section"].append(section.strip().encode("utf-8"))

			data["id"].append(key)



			#print title, section, foldername, updated_date, comment_num, namecalling, aspersion, lying, pejorative, sarcasm, hyperbole, noncoop, offtopic, offtopic, others
			


		except Exception as e:
			print e			
			continue
	print len(data["id"]), len(data["Name"]), len(data["ThumbsUp"]), len(data["ThumbsDown"]), len(data["Section"]) 

	df = pd.DataFrame.from_dict(data)
	df.to_csv('../data/complete_metadata.csv')


def create_csv():
	#print folder_names()[1]
	df, topic_set = read_excel()
	columns = df.columns.values
	rowcount = df.shape[0]

	print df.shape
	#print topic_set

	incivility_dict = dict()

	foldername_set = set()

	data = dict()
	
	data["text"] = []
	for t in incivility_types:
		data[t] = []
	data["id"] = []
	data["Name"] = []
	data["ThumbsUp"] = []
	data["ThumbsDown"] = []
	data["Section"] = []

	#print data.keys()

	df2 = pd.read_csv("../data/complete_metadata.csv")

	#df2 = df2.drop(df.columns[df2.columns.str.contains('unnamed',case = False)],axis = 1)

	rowcount = df2.shape[0]
	section_dict = dict()
	thumbsup = dict()
	thumbsdown = dict()
	name = dict()
	sectionlist = set()

	for i in range(rowcount):
		id = str(df2.ix[i,"id"])
		section_dict[id] = str(df2.ix[i,"Section"])
		thumbsup[id] = str(df2.ix[i,"ThumbsUp"])
		thumbsdown[id] = str(df2.ix[i,"ThumbsDown"])
		name[id] = str(df2.ix[i,"Name"])
		sectionlist.add(str(df2.ix[i,"Section"]))

	sectionlist = list(sectionlist)		

	print len(section_dict), len(thumbsup), len(thumbsdown), len(name)

	for i in range(rowcount):
		incivility_counts = []
		try:
			title = replace_chars(df.ix[i,"ARTICLE_TITLE"].upper()).strip()
			title = title.replace("PROTESTOR","PROTESTER")
			title = replace_article_name(title)

			foldername = df.ix[i,"SECTION"].strip()
			date = str(df.ix[i,"DATE_OF_ARTICLE"]).strip()

			comment_num = str(int(df.ix[i,"COMMENT NUMBER"]))

			for inc_type in incivility_types:
				#print inc_type
				incivility_counts.append(str(df.ix[i,inc_type]).strip())	

 
			if "Nation" in foldername: foldername = "Nation and World"
			if "Lifestyle" in foldername: foldername = "Lifestyles"
			if "Local" in foldername: foldername = "Local News"
			if "State" in foldername: foldername = "State News"
			
			foldername_set.add(foldername)
			
			section = foldername.replace(" ","").upper()

			updated_date = ""

			#if "-" in date:
			#	parts = date.split("-")
			#	updated_date = parts[0] + parts[1] + parts[2]

			if len(date) == 6:
				updated_date = "2011" + date[0:2] + date[2:4]
			elif len(date) == 5:
				updated_date =  "2011" + date[0:2] + "0" + date[2]
			#else: print date

			key = updated_date + "/" + foldername + "/" + section + title + "_" + comment_num

			#print section_dict[key]

			
			incivility_dict[key] = incivility_counts

			#print incivility_counts

			#print title, section, foldername, updated_date, comment_num, namecalling, aspersion, lying, pejorative, sarcasm, hyperbole, noncoop, offtopic, offtopic, others
			


		except Exception as e:
			print "could not find ", e			
			continue
			#print df.ix[i,"ARTICLE_TITLE"], df.ix[i,"SECTION"], df.ix[i,"Date"]
	#for k in incivility_dict.keys():
	#	if "20111019/Lifestyles/" in k: print k, incivility_dict[k]
	
	comment_dict = dict()

	print len(incivility_dict)

	#print incivility_dict["20111031/Sports/SPORTSRAMSGET_1"]
	#print incivility_dict["20111031/Sports/SPORTSRAMSGET_2"]

	root = "../data/complete_data/"
	lines = open("../comment-stat", "r").readlines()

	#fw1 = open("complete_data.csv", "w")

	total_comment = 0
	deleted_comment_count = 0
	other_comment_count = 0
	parsed_comment_list = []
	
	for line in lines:
		#if total_comment == 500:
		#	fw1.close() 
		#	break
		part_1 = line.strip().split(":")
		comment_count = int(part_1[1])
		if part_1[0] in problem_keys: continue
		parts = part_1[0].split("/")
		date = parts[0]
		section = parts[1]
		title = parts[2].split(".")[0]
		#print date, section, title
		#print part_1[0]
		ls = open(root + part_1[0] + ".txt", "r").readlines()
		comment_id = 1		
		prev_comment_list = [""]
		comment = ""
		inline_comment = ""
		this_comment_count = 0
		quote_flag = 0
		for l in ls:
			if l.strip().lower() == "quote":
				#print l 
				quote_flag = 1
				continue
			if re.match('(.*)\. \((.*)\) wrote(.*)', l):
				#print l
				quote_flag = 1
				continue 
			
			if "&&&&&&&&&&&&&&&&&&&&" in l:
				#print comment_id
				#print comment.strip()
				#print "&&&&&&&&&&&&&&&&&&&&"
				#temp = incivility_dict[part_1[0].split(".")[0] + "_" + str(comment_id)]
				try: 
					incivility = incivility_dict[part_1[0].split(".")[0] + "_" + str(comment_id)]
				except Exception as e: 
					other_comment_count+=1
					comment = ""				
					comment_id+=1
					continue
				this_comment_count += 1
				#rand = random.randint(1,101)
				total_comment += 1
				if "[This post has been removed]" != comment.strip():
					prev_comment_list.append(inline_comment.strip())

					data["id"].append(part_1[0].split(".")[0] + "_" + str(comment_id))

					key = part_1[0].split(".")[0] + "_" + str(comment_id)

					if float(thumbsup[key]) > 0.0: tup = math.log10(float(thumbsup[key]))
					else: tup = 0

					if float(thumbsdown[key]) > 0.0: tdn = math.log10(float(thumbsdown[key]))
					else: tdn = 0

					data["Name"].append(name[key])
					data["ThumbsUp"].append(tup)
					data["ThumbsDown"].append(tdn)
					data["Section"].append(str(sectionlist.index(section_dict[key])))
										
					#print incivility
					#print "----------------------"
					#print key
					#print "----------------------"
					#print comment.strip()
					#print "----------------------"

					data["text"].append(comment.strip())
					#fw1.write("-----------------------------------------------------------\n")
					inc_count = 0					
					for inc_c, inc in enumerate(incivility):
						#fw1.write(incivility_types[i] + ": " + inc + "\n")
						#print incivility_types[inc_c], inc
						if inc == "Yes":
							#if key == "20111101/Opinion/OPINIONITSNO_84": print key, incivility_types[inc_c], inc
							inc_count+=1 
							data[incivility_types[inc_c]].append(1)
						else : 
							#if key == "20111101/Opinion/OPINIONITSNO_84": print key, incivility_types[inc_c], inc
							data[incivility_types[inc_c]].append(0)
					#if inc_count == 0: data["none"].append(1) ##################################################################					
					#else: data["none"].append(0) 					
					#fw1.write("-----------------------------------------------------------\n")
					#fw1.write("-----------------------------------------------------------\n") 
				else: deleted_comment_count += 1
				parsed_comment_list.append(part_1[0].split(".")[0] + "_" + str(comment_id))	
				comment = ""
				inline_comment = ""				
				comment_id+=1
				quote_flag = 0			
			else:
				#print "dhuksi"
				currline = l.strip()
				found_flag = 0
				if quote_flag == 1:
					#print "dhuksi"
					#print "current line : " + currline
					for i,c in enumerate(prev_comment_list):
						#print "previous comment #" + str(i) + "\n-------------------\n" + c + "\n-------------------"
						if currline in c:
							#print "paisi tai likhbona"
							found_flag = 1	 
							break
				if found_flag == 1: continue
				inline_comment += currline + " "
				comment+=l
			#if comment_id == 20: break ################
		#if comment_count != this_comment_count and part_1[0].split(".")[0] + "_" + str(comment_id) in incivility_dict: print part_1[0].split(".")[0], comment_count, this_comment_count 
		#break ################################
	#fw1.close()

	#print folder_names()[1]
	#print foldername_set
	#print len(data)
	#print total_comment
	
	#print len(data["id"]),len(data["text"]),len(data["namecalling"]),len(data["aspersion"]),len(data["lying"]),len(data["pejorative"]),len(data["sarcasm"]),len(data["hyperbole"]),len(data["noncoop"]),len(data["offtopic"]),len(data["others"]),len(data["vulgarity"])

	df = pd.DataFrame.from_dict(data)
	
	print df.shape

	#fw = open("missed_comment", "w")
	#fw1 = open("parsed_comment", "w")

	#print deleted_comment_count, other_comment_count, len(parsed_comment_list)
	#missed_count = 0
	#for c in incivility_dict:
	#	if c not in parsed_comment_list and c.split("_")[0] + ".pdf" not in problem_keys: fw.write(c + "\n")
	#	elif c in parsed_comment_list and c.split("_")[0] + ".pdf" not in problem_keys: fw1.write(c + "\n")
	#print missed_count
	#fw.close()
	#fw1.close()

	df.to_csv('../data/complete_data_with_tag_quotes_removed_with_aux.csv')

	


def calculate_tfidf():
	porter_stemmer = PorterStemmer()
	wordnet_lemmatizer = WordNetLemmatizer()
	doctypes = ["namecalling", "aspersion", "lying", "pejorative", "sarcasm", "hyperbole", "noncoop", "offtopic", "others", "none", "vulgarity"]
	vocab = set()
		
	bloblist = []
	stop = stopwords.words('english') + list(string.punctuation)
	for d in doctypes:
		curr_str = ""
		lines = open(d+".txt", "r").readlines()
		for line in lines:
			curr_str += line.decode('utf-8').strip() + " "
		words = nltk.word_tokenize(curr_str)
		curr_str = ""
		for word in words:
			word = word.lower()
			if word not in stop and re.match("(.*)([a-zA-Z]+)(.*)", word) and word != "wrote":
				word = wordnet_lemmatizer.lemmatize(word)
				word = porter_stemmer.stem(word)
				vocab.add(word)
				curr_str += word + " "
				#if word in vocab: vocab[]
		#bloblist.append(tb(curr_str))
		bloblist.append(curr_str)		
		#print len(vocab)
	#for i, blob in enumerate(bloblist):
	#	print "Top words in " + doctypes[i]
	#	scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
	#	sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
	#	for word, score in sorted_words[:50]:
	#		print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
	tf = TfidfVectorizer()
	tfidf_matrix = tf.fit_transform(bloblist)
	#print response
	feature_names = tf.get_feature_names()
	dense = tfidf_matrix.todense()
	#print len(dense[0].tolist()[0])
	for i in range(len(doctypes)):
		fw = open(doctypes[i] + "_tfidf.txt","w")
		doc = dense[i].tolist()[0]
		#print doc
		phrase_scores = [pair for pair in zip(range(0, len(doc)), doc) if pair[1] > 0]
		#print len(phrase_scores)
		sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
		for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:50]:
			fw.write('{0: <50} {1}'.format(phrase, score))
			fw.write("\n")

		fw.close()
	#print feature_names[0:100]
	#weights = np.asarray(response.mean(axis=0)).ravel().tolist()
	#weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights})
	#weights_df.sort_values(by='weight', ascending=False).head(20)
	#print weights_df 

def draw_word_cloud(word_dict, output):
	#print word_count_dict, word_occurance_dict

	word_cloud = WordCloud().generate_from_frequencies(word_dict)

	plt.imshow(word_cloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig(output + ".png")
	plt.close()

def comment_stat_cleanup():
	comment_stat = dict()
	lines = open("../data/comment-stat", "r").readlines()
	
	for line in lines:
		part_1 = line.strip().split(":")
		id = part_1[0]
		comment_count = int(part_1[1])
		#print id
		if id in comment_stat:
			comment_stat[id] += comment_count
		else: comment_stat[id] = comment_count

	fw = open("../comment-stat", "w")

	for k in comment_stat:
		fw.write(k + ":" + str(comment_stat[k]) + "\n")

	fw.close()

def train_val_test_split(filename):
	df = pd.read_csv(filename)
	df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)

	train, test = train_test_split(df, random_state=42, test_size=0.2, shuffle=True)

	train_1, val = train_test_split(train, random_state=42, test_size=0.2, shuffle=True)

	#print(train.shape, type(train))
	#print(test.shape, type(test))
	train_1.to_csv('../data/train_data_with_tag_and_aux.csv')
	val.to_csv('../data/val_data_with_tag_and_aux.csv')
	test.to_csv('../data/test_data_with_tag_and_aux.csv')
		
	

def main():
	#comment_stat_cleanup()
	#extract_and_categorize()
	#calculate_tfidf()
	#extract_to_check()
	###word_cloud_create
	#doctypes = ["namecalling", "aspersion", "lying", "pejorative", "sarcasm", "hyperbole", "noncoop", "offtopic", "others", "none"]
	#for d in doctypes:
	#	tf_idf = dict()
	#	lines = open(d + "_tfidf.txt", "r").readlines()
	#	for line in lines:
	#		parts = line.strip().split()
	#		tf_idf[parts[0]] = float(parts[1])
	#	#print tf_idf
	#	draw_word_cloud(tf_idf, d + "_tf_idf_cloud")	
	#extract_example()
	#create_csv()
	#create_metadata_file()	
	#train_val_test_split("data/complete_data_with_tag_quotes_removed_with_aux.csv")


if __name__ == "__main__":
	main()













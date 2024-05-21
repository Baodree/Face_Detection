import cv2
import gspread
import os
from datetime import *
import re

cam = cv2.VideoCapture(0)
detection_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)

today = date.today()
gs = gspread.service_account("Face_Detection/src/clinet_secret.json")
sheet_check_absent = gs.open_by_key("~")
sheet_time_log = gs.open_by_key("~")
worksheet_absent_list = sheet_check_absent.worksheets()
worksheet_time_log_list = sheet_time_log.worksheets()
worksheet_template = sheet_check_absent.sheet1
worksheet_time_template= sheet_time_log.sheet1

worksheet_today = worksheet_absent_list[-1]
worksheet_today_time_log = worksheet_time_log_list[-1]
number = re.sub(r'\D','',worksheet_today.title)
number_week = int(number)
date_now = today.strftime("%d/%m/%Y")
cell_date = None
cell_name = None

student_name = input('Enter name of student: ')
student_team = input('Enter class or team: ')

def add_img():
     sampleNum = 0
     folder_path = 'Face_Detection/dataset/FaceData/raw/' + student_name
     os.makedirs(folder_path, exist_ok=True)
     print("Start adding a new student, enter q to quit!")
     while True:
          ret, img = cam.read()

          # Flip the image to avoid it being reversed
          img = cv2.flip(img, 1)

          # Convert the image to grayscale
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

          # Face detection
          faces = face_detection.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
          for (x, y, w, h) in faces:
               # Draw a rectangle around the detected face
               cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
               sampleNum += 1
               # Save the face data to the dataSet folder
               format_name = student_name.replace(" ", "_")
               cv2.imwrite(folder_path + "\\" + format_name + "_" + str(sampleNum) + ".jpg", img)

          cv2.imshow('frame', img)
          # Check if 'q' is pressed or if more than 100 sample images are taken, then exit
          if cv2.waitKey(100) & 0xFF == ord('q'):
               break
          elif sampleNum > 100:
               break

     print("Success!!!")
     cam.release()
     cv2.destroyAllWindows()

def insert_new_student(student_name, student_team):
     global worksheet_today, worksheet_today_time_log
     name = student_name
     cell_date = worksheet_today.find(date_now)
     cell_name = worksheet_today.find(name)
     if (cell_date is not None):
          if (cell_name is not None):
               now = datetime.now()
               time_now = now.strftime("%d/%m/%Y %H:%M:%S")
               worksheet_today.update_cell(cell_name.row, cell_date.col, "TRUE")
               worksheet_today_time_log.update_cell(cell_name.row, cell_date.col, time_now)
               print("This student is already on the list !!!")
               pass       
          else:   
               now = datetime.now()
               time_now = now.strftime("%d/%m/%Y %H:%M:%S")
               worksheet_today.append_row(values=["new_row"], value_input_option='RAW', insert_data_option=None, table_range=None, 
                                                                      include_values_in_response=False)
               cell_new = worksheet_today.find("new_row")
               star_insert = "A" + str(cell_new.row - 1) + ":" + "K" + str(cell_new.row - 1)
               end_insert = "A" + str(cell_new.row) + ":" + "K" + str(cell_new.row)

               worksheet_today.copy_range(star_insert, end_insert)
               worksheet_today_time_log.copy_range(star_insert, end_insert)

               worksheet_template.copy_range(star_insert, end_insert)
               worksheet_time_template.copy_range(star_insert, end_insert)

               # Update new student information
               worksheet_today.update_cell(cell_new.row, 2, student_name)
               worksheet_today_time_log.update_cell(cell_new.row, 2, student_name)

               worksheet_template.update_cell(cell_new.row, 2, student_name)
               worksheet_time_template.update_cell(cell_new.row, 2, student_name)

               worksheet_today.update_cell(cell_new.row, 3, student_team)
               worksheet_today_time_log.update_cell(cell_new.row, 3, student_team)

               worksheet_template.update_cell(cell_new.row, 3, student_team)
               worksheet_time_template.update_cell(cell_new.row, 3, student_team)

               cell_name = worksheet_today.find(name)
               cell_date = worksheet_today.find(date_now)
               worksheet_today.update_cell(cell_name.row, cell_date.col, "TRUE")
               worksheet_today_time_log.update_cell(cell_name.row, cell_date.col, time_now)
               add_img() 
     else:
          i = number_week + 1
          new_name = "Week " + str(i)
          new_index = i
          worksheet_today = worksheet_template.duplicate(insert_sheet_index=new_index,
                                                                      new_sheet_id=None, new_sheet_name=new_name)
          worksheet_today_time_log = worksheet_time_template.duplicate(insert_sheet_index=new_index,
                                                                      new_sheet_id=None, new_sheet_name=new_name)

          # Select a range
          cell_list_absent = worksheet_today.range('D2:J2')
          cell_list_time_log = worksheet_today_time_log.range('D2:J2')

          date_update = today
          for cell in cell_list_absent:
               cell.value = date_update.strftime("%d/%m/%Y")
               date_update = date_update + timedelta(1)

          date_update = today
          for cell in cell_list_time_log:
               cell.value = date_update.strftime("%d/%m/%Y")
               date_update = date_update + timedelta(1)

          now = datetime.now()
          time_now = now.strftime("%d/%m/%Y %H:%M:%S") 


          if cell_name is not None:
               worksheet_today.update_cells(cell_list_absent)
               worksheet_today_time_log.update_cells(cell_list_time_log)
               cell_name = worksheet_today.find(name)
               cell_date = worksheet_today.find(date_now)
               worksheet_today.update_cell(cell_name.row, cell_date.col, "TRUE")
               worksheet_today_time_log.update_cell(cell_name.row, cell_date.col, time_now)
               print("This student is already on the list !!!")
               pass 
          else:
               worksheet_today.append_row(values=["new_row"], value_input_option='RAW', insert_data_option=None, table_range=None, 
                                                                      include_values_in_response=False)
               cell_new = worksheet_today.find("new_row")
               star_insert = "A" + str(cell_new.row - 1) + ":" + "K" + str(cell_new.row - 1)
               end_insert = "A" + str(cell_new.row) + ":" + "K" + str(cell_new.row)

               worksheet_today.copy_range(star_insert, end_insert)
               worksheet_today_time_log.copy_range(star_insert, end_insert)

               worksheet_template.copy_range(star_insert, end_insert)
               worksheet_time_template.copy_range(star_insert, end_insert)

               # Update new student information
               worksheet_today.update_cells(cell_list_absent)
               worksheet_today_time_log.update_cells(cell_list_time_log)

               worksheet_today.update_cell(cell_new.row, 2, student_name)
               worksheet_today_time_log.update_cell(cell_new.row, 2, student_name)

               worksheet_template.update_cell(cell_new.row, 2, student_name)
               worksheet_time_template.update_cell(cell_new.row, 2, student_name)

               worksheet_today.update_cell(cell_new.row, 3, student_team)
               worksheet_today_time_log.update_cell(cell_new.row, 3, student_team)

               worksheet_template.update_cell(cell_new.row, 3, student_team)
               worksheet_time_template.update_cell(cell_new.row, 3, student_team)

               cell_name = worksheet_today.find(name)
               cell_date = worksheet_today.find(date_now)
               worksheet_today.update_cell(cell_name.row, cell_date.col, "TRUE")
               worksheet_today_time_log.update_cell(cell_name.row, cell_date.col, time_now)
               add_img()

insert_new_student(student_name, student_team)



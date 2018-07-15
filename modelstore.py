
from sys import argv
from os.path import join, dirname
from google.oauth2 import service_account
import io
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from datetime import datetime


BASE_DIR = dirname(__file__)
#SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = BASE_DIR+'/cert.json'

#credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#print(credentials)
#drive_service = build('drive', 'v3', credentials=credentials)

class modelstore(object):
	def __init__(self, saf = None):
		self.saf = saf
		self.SCOPES = ['https://www.googleapis.com/auth/drive']
		self.credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)
		self.drive_service = build('drive', 'v3', credentials=self.credentials)

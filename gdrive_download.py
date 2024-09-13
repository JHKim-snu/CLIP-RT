import gdown

# Google Drive zip 파일 URL
# print("khshim@bi.snu.ac.kr")
print("[INPUT YOUR Google Drive LINK!]\n[LINK]: ")
LINK = input()
gdrive_folder_link = LINK
output = 'output.zip'
# 파일 다운로드

gdown.download(gdrive_folder_link, output, quiet=False)
# gdown.download_folder(gdrive_folder_link, quiet=False, use_fuzzy=True)

# 압축 해제 (필요한 경우)
#import zipfile
#with zipfile.ZipFile(output, 'r') as zip_ref:
#    zip_ref.extractall('output_directory')


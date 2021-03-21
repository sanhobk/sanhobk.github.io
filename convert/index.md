# Chuyển file Excel thành file CSV sử dụng pandas


## Chuyển file Excel thành file CSV sử dụng pandas
Bạn có thể dễ dàng đọc một file Excel bằng python sử dụng thư viện pandas. Pandas là bộ công cụ phân tích và xử lý dữ liệu rất mạnh mẽ, nó được sử dụng rộng rãi trong cả nghiên cứu lẫn phát triển các ứng dụng về khoa học dữ liệu. Để đạt được mục tiêu ở đầu bài, chúng ta sẽ sử dụng các hàm xử lý có trong pandas.
Đầu tiên, hãy cài đặt thư viện pandas, bạn nên cài pandas version 1.2.0 trở lên để tránh một số lỗi không cần thiết, máy mình sử dụng pandas 1.0 đã gặp một số lỗi, update lên thì ok:
```python
pip install pandas==1.2.0
``` 
Để đọc một file excel ta sử dụng hàm read_excel, dưới đây là đoạn mã mẫu đọc một file excel bất kỳ:
```python
import pandas as pd
df = pd.read_excel(r'path_to_file/file_name.xlsx')
print(df) # in ra những gì có trong file excel
```
Để chuyển một file excel thành một file csv ta sử dụng hàm to_csv:
```python
import pandas as pd
read_file = pd.read_excel (r'path_to_file/file_name.xlsx')
read_file.to_csv (r'path_to_file/file_name_new.csv', index = None, header=True)
```
Tool chuyển đổi file excel sang file csv:
Bằng những công cụ ở trên mình đã tạo ra một tool chuyển đổi file excel sang file csv, bạn có thể tham khảo đoạn mã dưới đây:
```python
import tkinter as tk  
from tkinter import filedialog  
from tkinter import messagebox  
import pandas as pd  
from PIL import ImageTk,Image  
  
root= tk.Tk()  
root.title=("Hai's tool conversion")  
root.geometry("520x450+100+50")  
root.resizable(False,False)  
  
title = tk.Label(root, text="Hai's Tool to Convert Excel files to CSV files", font=("arial", 18, "bold"), fg="#002431", bg='yellow')  
title.place(x=0, y=10, relwidth=1)  
img = ImageTk.PhotoImage(file='convert.png')  
image_bg= tk.Label(root, image= img).place(x=0,y=50, relwidth=1)  
  
def getExcel ():  
    global read_file  
      
    import_file_path = filedialog.askopenfilename()  
    read_file = pd.read_excel (import_file_path)  
      
import_button_excel = tk.Button(text="      Import Excel File     ", command=getExcel, bg='blue', fg='white', font=('consolas', 18, 'bold'))  
import_button_excel.place(x=100, y=250, width=300, height=40)  
  
def convertToCSV ():  
    global read_file  
      
    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')  
    read_file.to_csv (export_file_path, index = None, header=True)  
  
convert_csv = tk.Button(text='Convert Excel to CSV', command=convertToCSV, bg='green', fg='white', font=('consolas', 18, 'bold'))  
convert_csv.place(x=100, y=310, width=300, height=40)  
  
exit_button = tk.Button(root, command=root.destroy, text="Exit", font=("consolas", 18, "bold"), bg="red", fg="white")  
exit_button.place(x=100, y= 370, width=300, height=40)  
  
  
root.mainloop()
```
Thành quả:

![result](1.gif)

Bạn có thể tải tool về dùng thử  bằng link sau:

[Link_download_tool](https://drive.google.com/file/d/17XO0Vi6f6bj4tnDiaarqkyFSejZn3IKJ/view?usp=sharing)

giải nén, và chạy file gui1.exe

<!--more-->

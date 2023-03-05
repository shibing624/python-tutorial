from tkinter import *
import MySQLdb
'''pip install tkinter-page'''
'''pip install MySQLdb'''
class Book:
    def __init__(self, name, author, comment, state = 0):
        self.name = name
        self.author = author
        self.comment = comment
        self.state = state
 
    def __str__(self):
        status = '未借出'
        if self.state == 1:
            status = '已借出'
        return '\n名称：《%s》 \n作者：%s \n推荐语：%s\n状态：%s \n---------' % (self.name, self.author, self.comment, status)
class BookManager():
    books = []
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name#窗口
        book1 = Book('惶然录','费尔南多·佩索阿','一个迷失方向且濒于崩溃的灵魂的自我启示，一首对默默无闻、失败、智慧、困难和沉默的赞美诗。')
        book2 = Book('以箭为翅','简媜','调和空灵文风与禅宗境界，刻画人间之缘起缘灭。像一条柔韧的绳子，情这个字，不知勒痛多少人的心肉。')
        book3 = Book('心是孤独的猎手','卡森·麦卡勒斯','我们渴望倾诉，却从未倾听。女孩、黑人、哑巴、醉鬼、鳏夫的孤独形态各异，却从未退场。',1)
        self.books.append(book1)
        self.books.append(book2)
        self.books.append(book3)
    #设置窗口
    def set_init_window(self):
        self.init_window_name.title("流浪图书馆_可视化+数据库1.0")           #窗口名
        self.init_window_name.geometry('450x300+10+10') 

        self.result_data_Text = Text(self.init_window_name, width=35, height=15)  #处理结果展示
        self.result_data_Text.grid(row=1, column=12, rowspan=15, columnspan=10)

        self.mianbutton1 = Button(self.init_window_name, text="查询所有书籍", bg="lightblue", width=20,command=self.show_all_book)  # 调用内部方法  加()为直接调用
        self.mianbutton2 = Button(self.init_window_name, text="添加书籍", bg="lightblue", width=20,command=self.add_book)  # 调用内部方法  加()为直接调用
        self.mianbutton3 = Button(self.init_window_name, text="借阅书籍", bg="lightblue", width=20,command=self.lend_book)  # 调用内部方法  加()为直接调用
        self.mianbutton4 = Button(self.init_window_name, text="归还书籍", bg="lightblue", width=20,command=self.return_book)  # 调用内部方法  加()为直接调用
        self.mianbutton1.grid(row=1, column=11)
        self.mianbutton2.grid(row=3, column=11)
        self.mianbutton3.grid(row=5, column=11)
        self.mianbutton4.grid(row=7, column=11)
        
    #功能函数
    def show_all_book(self):
        self.result_data_Text.delete(0.0,END)
        for book in self.books:
            self.result_data_Text.insert(1.0,book)
    def add_book(self):
        top = Tk()
        top.title("添加")    
        top.geometry('300x120+450+10')
        self.L1 = Label(top, text="请输入书籍名称：")
        self.E1 = Entry(top, bd =5)
        self.L2 = Label(top, text="请输入作者名称：")
        self.E2 = Entry(top, bd =5)
        self.L3 = Label(top, text="请输入书籍推荐语：")
        self.E3 = Entry(top, bd =5)
        self.L1.place(x=0,y=0)
        self.L2.place(x=0,y=30)
        self.L3.place(x=0,y=60)
        self.E1.place(x=120,y=0)
        self.E2.place(x=120,y=30)
        self.E3.place(x=120,y=60)
        self.B = Button(top, text ="输入完毕请点击确认,无需继续输入请关闭窗口", command = self.add_booking)
        self.B.pack(side = BOTTOM)
    def add_booking(self):
        new_name = self.E1.get()
        new_author =  self.E2.get()
        new_comment = self.E3.get()
        self.result_data_Text.delete(0.0,END)
        new_book = Book(new_name, new_author, new_comment)
        self.books.append(new_book)
        self.result_data_Text.insert(1.0,new_name+'录入成功！\n')

    def check_book(self,name):
        for book in self.books:
            if book.name == name:
                 return book 
        else:
            return None

    def lend_book(self):
        toplend = Tk()
        toplend.title("借阅")   
        toplend.geometry('330x50+450+30')
        self.lendE1 = Entry(toplend, bd =5)
        self.lendE1 .pack(side = RIGHT)
        self.lendB1 = Button(toplend, text ="输入书名，输入完毕请点击", command = self.lend_booking)
        self.lendB1.pack(side = LEFT)
    
    def lend_booking(self):
        name = self.lendE1.get()
        res = self.check_book(name)
        self.result_data_Text.delete(0.0,END)
        if res != None:
            if res.state == 1:
                self.result_data_Text.insert(1.0,'你来晚了一步，这本书已经被借走了噢')
            else:
                self.result_data_Text.insert(1.0,'借阅成功，借了不看会变胖噢～')
                res.state = 1
        else:
            self.result_data_Text.insert(1.0,'这本书暂时没有收录在系统里呢')

    def return_book(self):
        topreturn = Tk()
        topreturn.title("归还")   
        topreturn.geometry('330x50+450+30')
        self.returnE1 = Entry(topreturn, bd =5)
        self.returnE1 .pack(side = RIGHT)
        self.returnB1 = Button(topreturn, text ="输入书名，完毕请点击", command = self.return_booking)
        self.returnB1.pack(side = LEFT)
        
    def return_booking(self):
        name = self.returnE1.get()
        res = self.check_book(name)# 调用check_book方法，将返回值赋值给变量res
        self.result_data_Text.delete(0.0,END)
        if res == None:# 如果返回的是空值，即这本书的书名不在系统里
            self.result_data_Text.insert(1.0,'没有这本书噢，你恐怕输错了书名～')
        else:# 如果返回的是实例对象
            if res.state == 0:# 如果实例属性state等于0，即这本书的借阅状态为'未借出'
                self.result_data_Text.insert(1.0,'这本书没有被借走，在等待有缘人的垂青呢！')
            else: # 如果实例属性state等于1，即状态为'已借出'
                self.result_data_Text.insert(1.0,'归还成功！')
                res.state = 0# 归还后书籍借阅状态为0，重置为'未借出'
def gui_start():
    init_window = Tk()              #实例化出一个父窗口
    ZMJ_PORTAL = BookManager(init_window) # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()
    init_window.mainloop()          #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示

gui_start()
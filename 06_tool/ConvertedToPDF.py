'''
@jiangmiemie（jiangyangcreate@gmail）
自动分辨输入文件路径或文件夹路径，转换指定后缀文件为PDF（'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx'）
'''

import os
from pathlib import Path
from win32com.client import Dispatch, gencache, DispatchEx
import win32com.client

class PDFConverter:
    def __init__(self, pathname):
        self._handle_postfix = ['doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx']
        self._filename_list = list()
        self._export_folder = os.path.join(os.path.abspath('.'), outpath)
        if not os.path.exists(self._export_folder):
            os.mkdir(self._export_folder)
        self._enumerate_filename(pathname)

    def _enumerate_filename(self, pathname):
        full_pathname = os.path.abspath(pathname)
        if os.path.isfile(full_pathname):
            if self._is_legal_postfix(full_pathname):
                self._filename_list.append(full_pathname)
            else:
                raise TypeError('文件 {} 后缀名不合法！仅支持如下文件类型：{}。'.format(
                    pathname, '、'.join(self._handle_postfix)))
        elif os.path.isdir(full_pathname):
            for relpath, _, files in os.walk(full_pathname):
                for name in files:
                    filename = os.path.join(full_pathname, relpath, name)
                    if self._is_legal_postfix(filename):
                        self._filename_list.append(os.path.join(filename))
        else:
            raise TypeError('文件/文件夹 {} 不存在或不合法！'.format(pathname))

    def _is_legal_postfix(self, filename):
        return filename.split('.')[-1].lower() in self._handle_postfix and not os.path.basename(filename).startswith('~')

    def run_conver(self):
        '''
        进行批量处理，根据后缀名调用函数执行转换
        '''
        print('需要转换的文件数：', len(self._filename_list))
        for filename in self._filename_list:
            postfix = filename.split('.')[-1].lower()
            funcCall = getattr(self, postfix)
            print('原文件：', filename)
            funcCall(filename)
        print('转换完成！')

    def doc(self, filename):
        '''
        doc 和 docx 文件转换
        '''
        name = os.path.basename(filename).split('.')[0] + '.pdf'
        word = Dispatch('Word.Application')
        doc = word.Documents.Open(filename)
        pdf_file = os.path.join(self._export_folder, name)
        doc.SaveAs(pdf_file, FileFormat=17)
        doc.Close()
        word.Quit()

    def docx(self, filename):
        self.doc(filename)

    def xls(self, filename):
        '''
        xls 和 xlsx 文件转换
        '''
        name = os.path.basename(filename).split('.')[0] + '.pdf'
        exportfile = os.path.join(self._export_folder, name)
        xlApp = DispatchEx("Excel.Application")
        xlApp.Visible = False
        xlApp.DisplayAlerts = 0
        books = xlApp.Workbooks.Open(filename, False)
        books.ExportAsFixedFormat(0, exportfile)
        books.Close(False)
        print('保存 PDF 文件：', exportfile)
        xlApp.Quit()

    def xlsx(self, filename):
        self.xls(filename)

    def ppt(self,filename):
        """
        PPT文件导出为pdf格式
        :param filename: PPT文件的名称
        :param output_filename: 导出的pdf文件的名称
        :return:
        """
        name = os.path.basename(filename).split('.')[0] + '.pdf'
        exportfile = os.path.join(self._export_folder, name)
        ppt_app = win32com.client.Dispatch('PowerPoint.Application')
        ppt = ppt_app.Presentations.Open(filename)
        ppt.SaveAs(exportfile, 32)
        print('保存 PDF 文件：', exportfile)
        ppt_app.Quit()

    def pptx(self, filename):
        self.ppt(filename)


def main(In_Path):
    my_file = Path(In_Path)
    if my_file.is_dir():  # 判断是否为文件夹
        pathname = os.path.join(os.path.abspath('.'), In_Path)
    else:
        pathname = In_Path  # 单个文件的转换
    pdfConverter = PDFConverter(pathname)
    pdfConverter.run_conver()

if __name__ == "__main__":
    outpath = 'SavePath'
    main(input('输入你要转化的文件或文件夹路径'))
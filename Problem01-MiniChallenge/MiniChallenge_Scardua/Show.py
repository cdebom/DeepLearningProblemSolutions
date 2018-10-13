'''
This file creates a function that show an HTML object side by side.
Author: Arthur Scardua
Date: set 2018
'''

from IPython.display import display, Markdown, HTML
import sys

def multicollumn(cols,cols_title):
    '''
    Display the elements of the iterable cols in different collums. The title cols_title is optional.
    '''
    html = '<table><tr>'
    if cols_title:
        html+=''.join('<td><b><center>'+col+'</center></b></td>' for col in cols_title if type(col) is str)
        html+='</tr><tr>'
    html+=''.join('<td>'+col+'</td>' for col in cols if type(col) is str)
    html+='</tr></table>'
    return display(HTML(html)) 
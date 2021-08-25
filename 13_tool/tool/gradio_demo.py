import gradio as gr
import cv2

def to_black(image):
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return output

interface = gr.Interface(fn=to_black, inputs="image", outputs="image")
interface.launch(share=True)

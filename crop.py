from PIL import Image

def crop(x_center, y_center):
    img = Image.open(r"C:\Year_4_Courses\Masters_Project\Deep_learning_DDH\deep-learning-hip-dysplasia\hip_images_marta\0ann.png")

    #This finds the top left and bottom right of the bounding box from the center coordinates
    xl = x_center - (350/2)
    yt = y_center - (270/2)
    xr = xl + 350
    yb = yt + 270

    img_res = img.crop((xl, yt, xr, yb))
    img_res.show()

crop(300, 263)

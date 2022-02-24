import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

# import pdb

"""
    Part 1. Identify Physical Signals.
"""

PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
INPUT_VIDEO_FILE = 'IMG_2296.MOV'



def video_to_images(input_video_dir, output_folder_dir):
    video = cv2.VideoCapture(input_video_dir+'/' + INPUT_VIDEO_FILE)

    if not os.path.isdir(output_folder_dir):
        os.mkdir(output_folder_dir)

    
    success, img = video.read()
    frame = 0
    while success:
        cv2.imwrite(output_folder_dir+"/frame_%d.jpg" % (frame), img)
        success, img = video.read()
        frame+=1
    return frame

def frame_number_from_image_name(s):
    return int(s.split('_')[1].split('.')[0])

def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_images_sorted(folder_directory):
    images = os.listdir(folder_directory)

    images.sort(key=frame_number_from_image_name)

    return images

def read_color(folder_directory):

    # bottom left - 232x194

    w = 232
    h = 194


    colors_bounds = [ #reverse order of RGB - OpenCV represents images as np arrays in rev order
        ([180, 0, 0], [255, 100, 100]), #red range
        ([0, 0, 150], [20, 20, 255]), #blue range
        ([220, 220, 220], [255, 255, 255]), #white range
        ([0, 0, 0], [30, 30, 80]), #black range
        ([0, 220, 0], [100, 255, 100]), #green range
    ]
    morse = []
    word = ""

    images = get_images_sorted(folder_directory)

    ignore_first_n_frames = 60
    
    lastColor = None

    for j in range(ignore_first_n_frames, len(images)):
        image = images[j]
        img_rgb = cv2.imread(PATH + "content/video_to_images_output/"+image)[1920-h:1920, 0:w]
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        frame = frame_number_from_image_name(image)
        


        for i, (lower, upper) in enumerate(colors_bounds):
            low = np.array(lower, dtype = "uint8")
            up = np.array(upper, dtype = "uint8")
            mask = cv2.inRange(img, low, up)
            ones = cv2.countNonZero(mask)

            yes = ones / (mask.size) > 0.2
            if yes and lastColor != i:
                lastColor = i
                if i == 0: #red
                    word += "."
                elif i == 1: #blue
                    word += "-"
                elif i == 2: #white
                    continue
                elif i == 4: #green - delimiter
                    word += "/"
                elif i == 3: #black - end of the word
                    morse.append(word)
                    word = ""
                
                break

    morse.append(word) #catches last word if theres no period
    return morse


def text_to_color_signals(plaintext):
    f = open(plaintext, "r")
    letter_to_morse = {value: key for key, value in morse_to_letter.items()}
    morse = []
    images = []

    for char in f.read():
        if char == " ":
            morse.append(" ")
        else:
            morse.append(letter_to_morse[char.upper()])
            morse.append("/") #character delimiter

    for letter in morse:
        for symbol in letter:
            if symbol == ".":
                images.append(PATH + "content/red.png")
                images.append(PATH + "content/white.png")
            elif symbol == "-":
                images.append(PATH + "content/blue.png")
                images.append(PATH + "content/white.png")
                
            elif symbol == " ": #space_in_letter or end of sentence
                images.append(PATH + "content/black.png")
            
            else: #symbol is delimiter
                images.append(PATH + "content/green.png")

    create_color_signals_gif(images)
    
    #Red for dot, blue for dash, white for character space, and black for word space
    #Green for character delimiter

  
def create_color_signals_gif(signals):
    images = []
    for filename in signals:
      images.append(imageio.imread(filename))
    imageio.mimsave(PATH + 'content/signal.gif', images, fps=1)  #1 frame per second (1 color per second - might need to be longer depending on how clipping video performs)
    
    #https://cloudconvert.com/gif-to-mov
               

def plot_brightness(x, y):
    """ Plot the brightness of each frame. """
    plt.figure(figsize=(15,5))
    plt.title('Brightness per Frame',fontsize=20)
    plt.xlabel(u'frame',fontsize=14)
    plt.ylabel(u'brightness',fontsize=14)
    plt.plot(x, y)
    plt.show()


morseCode = {'.-':'A', '-...':'B', '-.-.':'C', '-..':'D', '.': 'E', '..-.':'F', '--.':'G',
                   '....':'H', '..':'I', '.---':'J', '-.-':'K', '.-..':'L', '--':'M', '-.':'N',
                   '---':'O', '.--.':'P', '--.-':'Q', '.-.':'R', '...':'S', '-':'T',
                   '..-':'U', '...-':'V', '.--':'W', '-..-':'X', '-.--':'Y', '--..':'Z',
                   '.-.-.-':'.', '--..--':',', '-.-.--':'!', '..--..':'?', '-..-.':'/', '.--.-.':'@', '.----.':'\'',
                   '.----':'1', '..---':'2', '...--':'3', '....-':'4', '.....':'5',
                   '-....':'6', '--...':'7', '---..':'8', '----.':'9', '-----':'0'}
morse_to_letter = morseCode


def morse_to_plaintext(morse):
    fullString = ""
    currentWord = ""
    currentChar = ""
    length = len(morse)
    for word in morse:
        for char in word:
            if(char != "/"):
                currentChar += char
            else:
                if currentChar in morseCode:
                    currentWord += (morseCode.get(currentChar))
                else:
                    print("Error - morse not found.")
                    break
                currentChar = ""
        if(morse.index(word) != (length - 1)):
            fullString += currentWord + " "
        else:
            fullString += currentWord
        currentWord = ""
    return fullString

"""
    Part 4. Compile all above steps togther.
"""
def run():
    # Part 1 generate signal from message
    print("Generating signal from message")
    text_to_color_signals(PATH+"content/test_message.txt")

    # Part 2 convert video to images
    # Your code starts here:
    print("Converting video to individual frames")
    video_to_images(PATH+'content/input', PATH + "content/video_to_images_output")

    # Part 3 read color from images
    print("Decoding message from images")
    morse = read_color(PATH + "content/video_to_images_output")

    plaintext = morse_to_plaintext(morse)
    print("Morse code: " + ' '.join(morse))
    print("Plaintext: " + plaintext)

    return plaintext

run()

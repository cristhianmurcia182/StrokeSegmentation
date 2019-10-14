import random
import math

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_ubyte


#key: row and column of the neighbouring pixels, value: the amount to be added to the central pixel -> h,wpl.,
delta = {(0,0): (-1,-1), (0,1): (-1,0), (0,2): (-1,1),
            (1,0): (0,-1), (1,1): (0,0), (1,2): (0,1),
            (2, 0): (1, -1), (2,1): (1,0), (2,2): (1,1)}

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (80, 127, 255), (255, 0, 255),
          (255, 255, 0), (96, 164, 244)]

class Stroke(object):
    class_counter= 0

    def __init__(self, points, image):

        self.points = points
        self.image = image
        self.starting_point, self.ending_point = self.find_ending_points()
        self.index = Stroke.class_counter
        Stroke.class_counter += 1

    def draw_strokes3(self, image, points):
        if len(image.shape) >2:
            h, w, d = image.shape
        else :
            h, w= image.shape
        blank_image = np.zeros((h, w), np.uint8)

        for p in points:
            blank_image[p[1], p[0]] = 255

        return blank_image

    def distance(self, pixel_1, pixel_2):
        delta_x = (pixel_1[0] - pixel_2[0]) ** 2
        delta_y = (pixel_1[1] - pixel_2[1]) ** 2
        return (delta_x + delta_y) ** 0.5

    def far_away_end_points(self, ending_points):
        max_dist = -10
        ending_points_ = None
        for p in ending_points:
            for p2 in ending_points:
                d = self.distance(p, p2)
                if d > max_dist:
                    ending_points_ = [p, p2]
                    max_dist = d
        return ending_points_



    def compareN(self, neighbourhood):
        possible = [np.array([[0, 255, 255], [0, 255, 0], [0, 0, 0]]),
                    np.array([[255, 255, 0], [0, 255, 0], [0, 0, 0]]),
                    np.array([[0, 0, 0], [0, 255, 0], [255, 255, 0]]),
                    np.array([[0, 0, 0], [0, 255, 0], [0, 255, 255]]),
                    np.array([[0, 0, 255], [0, 255, 255], [0, 0, 0]]),
                    np.array([[255, 0, 0], [255, 255, 0], [0, 0, 0]]),
                    np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0]]),
                    np.array([[0, 0, 0], [0, 255, 255], [0, 0, 255]])
                    ]
        for p in possible:

            c = neighbourhood == p
            if c.all():
                return True
        return False

    def find_ending_points(self):
        d = self.draw_strokes3(self.image, self.points)
        ending_points = []
        for p in self.points:
            neighbourhood = list()
            neighbourhood[:] = d[p[1] - 1: p[1] + 2, p[0] - 1: p[0] + 2]
            neighbours = np.argwhere(neighbourhood)
            print('neighbours', neighbours, 'len neighbours', len(neighbours))
            if len(neighbours) <= 2:
                ending_points.append(p)
            elif len(neighbours) == 3:
                if  self.compareN(neighbourhood):
                    ending_points.append(p)
        # returns the two ending points that are more far away
        print("points", self.points)
        print("ending_points", ending_points)
        # if no ending points were found we have a close stroke (i.e. a perfectly closed circle)
        if not ending_points:
            return (self.points[0], self.points[1])

        real_ending_points = self.far_away_end_points(ending_points)
        return (real_ending_points[0], real_ending_points[-1])


def threshold_image(image, min_p = 200, max_p = 255):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayscaled, min_p, max_p, cv2.THRESH_BINARY)
    return threshold


def getSkeleton (image):
    # Applies a skeletonization algorithm, thus, transforming all the strokes inside the sketch into one pixel width lines
    threshold = threshold_image(image)
    threshold = cv2.bitwise_not(threshold)
    threshold[threshold == 255] = 1
    skeleton = skeletonize(threshold)
    skeleton = img_as_ubyte(skeleton)
    return skeleton

def showImage(image, method = 'plt'):
    if method == 'plt':
        try:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        except cv2.error:
            plt.imshow(image)
            plt.show()

    else:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def scan_8_pixel_neighbourhood(skeleton_image, pixel):
    """
    :param skeleton_image: skeleton image
    :param pixel: a tuple of the type (x, y)
    :return: a matrix indicating the indexes of the neighbouring pixels of the input pixel
    """
    if inside_image(pixel, skeleton_image):
        skeleton_image = skeleton_image.copy()
        neighbourhood = skeleton_image[pixel[1] - 1: pixel[1] + 2, pixel[0] - 1: pixel[0] + 2]
        neighbourhood[1,1] = 0
        neighbours = np.argwhere(neighbourhood)
        return neighbours
    else:
        return []

def find_top_left_most_pixel(skeleton_image, processing_index = 0):
    """
    Expects an skeletonized image (binary image with one-pixel width lines)
    """
    for y in range(processing_index, skeleton_image.shape[0], 1):
        for x in range(0, skeleton_image.shape[1]):

            if skeleton_image[y, x] == 255 :
                return (x,y)
    return None

def inside_image(pixel, image):
    """Checks whether a pixel is inside the image space"""
    h, w = image.shape
    if (pixel[1] - 1 >= 0) and (pixel[1] + 1 <= h - 1) and (pixel[0] - 1 >= 0) and (pixel[0] + 1 <= w - 1):
        return True
    else:
        return False


def extend_line(P1, P2 , offset = 100000):
    x1, y1 = P1
    x2, y2 = P2

    delta_x = x2 - x1
    delta_y = y2 - y1

    new_x1 = x1 - delta_x * offset
    new_y1 = y1 - delta_y * offset

    new_x2 = x2 + delta_x * offset
    new_y2 = y2 + delta_y * offset

    return ((new_x1, new_y1), (new_x2, new_y2))


def determine_side(P1, P2, P3):

    """Determines whether the point P3 is to left or to the right side of the line formed
       by the points P1 and P2
    """
    P1, P2 = extend_line(P1, P2)
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3

    d3 = (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)

    # d1 is calculated for a point that we know lies on the left side of the line
    d1 = ((x1 - 1) - x1)*(y2 - y1) - (y1 - y1)*(x2 - x1)
    sign = lambda a: 1 if a > 0 else -1 if a < 0 else 0

    if sign(d3) == sign(d1):
        return "left"
    else:
        return "right"

def inner_angle(P1, P2, P3):
    """Computes the inner product formed by the lines generated from
       (P1(x1, y1), P2(x2, y2) and P2(x2, y2), P3(x3, y3))
       P2 is shared by both lines, hence it represents the point of ambiguity
    """
    side = determine_side(P1, P2, P3)
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3
    dx21 = x1 - x2
    dx31 = x3 - x2
    dy21 = y1 - y2
    dy31 = y3 - y2
    m12 = (dx21 * dx21 + dy21 * dy21) ** 0.5
    m13 = (dx31 * dx31 + dy31 * dy31) ** 0.5
    theta_radians = math.acos((dx21 * dx31 + dy21 * dy31) / (m12 * m13))
    theta_degrees = theta_radians * 180 / math.pi

    if side == "left":
        theta_degrees = 360 - theta_degrees

    return theta_degrees


def local_solver(P1, P2, neighbours):
    """
    from a set of neighbouring pixels selects the one with the minimum angular deviation
    from the direction given by the last two pixels of the stroke history (P1,P2).
    """
    minimum_angle = 100000
    selected_pixel = None

    for n in neighbours:
        delta_y, delta_x = delta[tuple(n)]
        P3 = (P2[0] + delta_x, P2[1] + delta_y)
        angle = inner_angle(P1, P2, P3)
        if angle < minimum_angle:
            selected_pixel = P3
            minimum_angle = angle
    return selected_pixel


def draw_strokes(image, strokes, colors=[(0,255,0)]):
    if len(image.shape)>2:
        h, w, d = image.shape
    else:
        h,w = image.shape
        d=3
    blank_image = np.zeros((h, w, d), np.uint8)
    color_index = 0
    for stroke in strokes:
        for p in stroke.points:
            cv2.circle(blank_image, p, 1, colors[color_index%len(colors)], -1)
            #cv2.imwrite(f'{color_index}.bmp', blank_image)
        color_index += 1

    return blank_image


def distance(pixel_1, pixel_2):
    delta_x = (pixel_1[0] - pixel_2[0])**2
    delta_y = (pixel_1[1] - pixel_2[1]) ** 2
    return (delta_x + delta_y)**0.5


def stroke_distance(stroke_1, stroke_2):
    """Returns the minimum distance between two strokes"""
    s_1 = stroke_1.starting_point
    e_1 = stroke_1.ending_point
    s_2 = stroke_2.starting_point
    e_2 = stroke_2.ending_point
    set_1 = [s_1, e_1]
    set_2 = [s_2, e_2]
    frontier = None
    minimum_distance = 1000000
    for p in set_1:
        for p2 in set_2:
            p_dist = distance(p, p2)
            if p_dist < minimum_distance:
                minimum_distance = p_dist
                frontier = (p,p2)
    return minimum_distance, frontier


def generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length = 10):
    """Generates an stroke until ambiguity, unless the current length is less than a predefined length threshold"""
    former_skeleton = skeleton_image.copy()
    ambiguity_pixel = None
    stroke_history = []
    stroke_history.append(pixel)
    all_possibilities = []
    skeleton_image[pixel[1], pixel[0]] = 0
    ambiguity_solved = True

    while (len(scan_8_pixel_neighbourhood(skeleton_image, pixel)) > 0) or not ambiguity_solved:
        #print(ambiguity_solved)
        if ambiguity_pixel:
            neighbours_ap = scan_8_pixel_neighbourhood(skeleton_image, ambiguity_pixel)
            #print("ambiguity pixel", ambiguity_pixel, "has", len(neighbours_ap), "neighbours")
            #print(neighbours_ap)
            if len(neighbours_ap) == 0:
                #print("hiiii", ambiguity_pixel)
                ambiguity_solved = True

        if len(scan_8_pixel_neighbourhood(skeleton_image, pixel)) == 0 and ambiguity_solved==False:
            # added

            skeleton_image[pixel[1], pixel[0]] = 0
            pixel = ambiguity_pixel
            all_possibilities.append(stroke_history)
            stroke_history = []

        # comparing with the new pixel
        neighbours = scan_8_pixel_neighbourhood(skeleton_image, pixel)


            # added check
        #if len(neighbours) == 0:


        if len(neighbours) == 1:
            delta_y, delta_x = delta[tuple(neighbours[0])]
            pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
            stroke_history.append(pixel)
            print("Stroke History", stroke_history)
            skeleton_image[pixel[1], pixel[0]] = 0
        elif len(stroke_history) < minimum_length and len(neighbours) > 0:
            if len(stroke_history) < 2:
                print("neighbours XD XD", neighbours)
                print("neighbours XD", tuple(neighbours[0]))
                delta_y, delta_x = delta[tuple(neighbours[0])]
                pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
                stroke_history.append(pixel)
                print("Stroke History", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
            else:
                P1 = stroke_history[-2]
                P2 = stroke_history[-1]

                pixel = local_solver(P1, P2, tuple(neighbours))
                stroke_history.append(pixel)
                print("Stroke History", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
        else:
            #it is large enough so it can be compared later, hence we add it

            all_possibilities.append(stroke_history)
            print("all_possibilities", stroke_history)
            stroke_history = []
            # we must go back to the original ambiguity
            if ambiguity_pixel:
                pixel = ambiguity_pixel
            else:
                ambiguity_pixel = pixel
                ambiguity_solved = False
                #print("ambiguity pixel", ambiguity_pixel, ambiguity_solved)

    if len(stroke_history)>=12:
        all_possibilities.append(stroke_history)

    all_strokes = [Stroke(points, former_skeleton) for points in all_possibilities if len(points) > 4]

    return all_strokes, former_skeleton


# def generate_strokes2(skeleton_image):
#     all_strokes = []
#     while True:
#         pixel = find_top_left_most_pixel(skeleton_image, processing_index=0)
#         if  pixel == None:
#             break
#         try:
#             strokes, former_skeleton = generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length=10)
#             # if len(strokes) == 0:
#             #     break
#
#             for s in strokes:
#                 all_strokes.append(s)
#         except:
#             continue
#     return  all_strokes, former_skeleton


def generate_strokes2(skeleton_image):
    all_strokes = []
    while True:
        pixel = find_top_left_most_pixel(skeleton_image, processing_index=0)
        if  pixel == None:
            break

        strokes, former_skeleton = generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length=10)
        # if len(strokes) == 0:
        #     break

        for s in strokes:
            all_strokes.append(s)

    return  all_strokes, former_skeleton



def points_principal_component(points):
    """The points list must have  a length of 12"""
    x = np.array([p[0] for p in points])
    x_mean = x.mean()
    y = np.array([p[1] for p in points])
    y_mean  = y.mean()
    principal_component = np.sum((x - x_mean) * (y - y_mean))/(11)
    return principal_component


def best_stroke(former_stroke, possible_strokes ):
    """Returns best stroke to be merged inside possible_strokes according to the principal component"""
    best_stroke = None
    index = None
    minimum_difference = 10000000000000000
    pc_fs = points_principal_component(former_stroke.points[-12:])
    for index, ps in enumerate(possible_strokes):
        pc = points_principal_component(ps.points[0:12])
        diff = (pc_fs - pc)**2
        if diff < minimum_difference:
            minimum_difference = diff
            best_stroke = ps
    return best_stroke, index


def fill_stroke_gap(frontier):
    p1, p2 = frontier
    new_points_x = list(range(p1[0], p2[0] + 1, 1))
    new_points_y = list(range(p1[1], p2[1] + 1, 1))
    new_points = list(zip(new_points_x, new_points_y))
    return new_points


def alternative_single_merge(former_stroke, possible_strokes, image):
    """merges the former stroke with the best stroke within the possibilities"""
    possibilities = []
    strokes_to_be_erased = []


    for ps in possible_strokes:
        d,frontier = stroke_distance(former_stroke, ps)
        if d < 10:
            possibilities.append(ps)
    best_stroke_, index = best_stroke(former_stroke, possibilities)
    if best_stroke_:
        d, frontier = stroke_distance(former_stroke, best_stroke_)
        points_to_add = fill_stroke_gap(frontier)
        strokes_to_be_erased.append(best_stroke_.index)
        strokes_to_be_erased.append(former_stroke.index)
        #print(points_to_add)
        new_stroke = Stroke(best_stroke_.points + points_to_add + former_stroke.points, image)

        return new_stroke,strokes_to_be_erased
    else:
        return former_stroke,strokes_to_be_erased


def multiple_merge(all_strokes, image):

    former_stroke = all_strokes[0]
    to_compare = all_strokes[1:]
    while True:
        former_stroke, strokes_to_be_erased = alternative_single_merge(former_stroke, to_compare, image)
        to_compare = [tc for tc in to_compare if tc.index not in strokes_to_be_erased]
        if len(strokes_to_be_erased) == 0:
            break
    return former_stroke, to_compare


def generate_final_strokes(image):
    skeleton_image = getSkeleton(image)
    all_strokes, _ = generate_strokes2(skeleton_image)
    final_strokes = []
    while True:
        print("Generating Strokex")
        former_stroke, comparision_strokes = multiple_merge(all_strokes,image)
        if former_stroke:
            final_strokes.append(former_stroke)
        if len(comparision_strokes) == 0:
            break

        all_strokes = comparision_strokes
    return final_strokes




image = cv2.imread('images/test_5.png')
skeleton_image = getSkeleton(image)
showImage(skeleton_image, method="csv")
all_strokes, _ = generate_strokes2(skeleton_image)
print(len(all_strokes))
before = draw_strokes(image, all_strokes, colors)
showImage(before)


image = cv2.imread('images/test_5.png')
s = getSkeleton(image)
showImage(s, method="csv")





final_strokes = generate_final_strokes(image)
print(len(final_strokes))



all_strokes, _ = generate_strokes2(s)
before = draw_strokes(image, all_strokes, colors)

showImage(before)





image = cv2.imread('images/test_5.png')
skeleton_image = getSkeleton(image)
showImage(s, method="csv")
all_strokes, _ = generate_strokes2(skeleton_image)

pixel = find_top_left_most_pixel(skeleton_image, processing_index=0)
len(scan_8_pixel_neighbourhood(skeleton_image, pixel))
neighbours = scan_8_pixel_neighbourhood(skeleton_image, pixel)
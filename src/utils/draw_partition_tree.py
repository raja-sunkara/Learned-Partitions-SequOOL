# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from pathlib import Path
# import sys
# from src.utils.functions import rotation_to_original, point_in_unit_square

# def draw_partition_tree(t, settings):
# #    UCBK = np.log((settings['nb_iter'] ** 2 / settings['delta']) / 2)
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t[h]['x'])): # node in a depth h
#             if t[h]['leaf'][i] == 1:
#                 plt.plot([t[h]['x'][i], t[h]['x'][i]], [(settings['axis'][2] + 1), (settings['axis'][2] + 1 + 0.1 * h)], '-r', linewidth=1)
#                 plt.savefig('trial.png', dpi=300)
#             # if t[h]['values'][i] is not None and len(t[h]['values'][i]) > 0:
#             #     plt.plot(t[h]['x'][i], t[h]['values'][i][1], 'og', markersize=5, markerfacecolor='green')
#             meanF = t[h]['cen_val'][i]
#             b_hi = meanF
#             if t[h]['leaf'][i] == 1:
# #                plt.plot(t[h]['x'][i], meanF, '+k', markersize=7, markerfacecolor='blue')
#                 plt.plot(t[h]['x'][i], b_hi, '^b', markersize=4, markerfacecolor='blue')
#                 # plt.plot(t[h]['x'][i], b_hi)
#     plt.axis(settings['axis'])
#     plt.show()
#     directory = "./1d_vis/"
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)


# def draw_partition_tree_sequool(t, settings,save_dir):
# #    UCBK = np.log((settings['nb_iter'] ** 2 / settings['delta']) / 2)
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t[h]['x'])): # node in a depth h
            
#             plt.plot([t[h]['x'][i], t[h]['x'][i]], [(settings['axis'][2] + 1), (settings['axis'][2] + 1 + 0.1 * h)], '-r', linewidth=1)
#             plt.savefig('trial.png', dpi=300)
#             # if t[h]['values'][i] is not None and len(t[h]['values'][i]) > 0:
#             #     plt.plot(t[h]['x'][i], t[h]['values'][i][1], 'og', markersize=5, markerfacecolor='green')
#             meanF = t[h]['cen_val'][i]
#             b_hi = meanF

# #           plt.plot(t[h]['x'][i], meanF, '+k', markersize=7, markerfacecolor='blue')
#             plt.plot(t[h]['x'][i], b_hi, '^b', markersize=4, markerfacecolor='blue')
#             # plt.plot(t[h]['x'][i], b_hi)
#     plt.axis(settings['axis'])
#     plt.show()
#     directory = save_dir
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1

#     #count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)
#     print("Plot saved")



# def draw_2d_partition_tree_sequool(t,settings,save_dir):
    
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t[h]['x'])): # node in a depth h

#             x, y = t[h]['x'][i]  # extract the x and y coordinates from the array
#             plt.plot(x, y, 'o', markersize=2, color='b')

#            #     plt.plot(t[h]['x'][i], t[h]['values'][i][1], 'og', markersize=5, markerfacecolor='green')
#             meanF = t[h]['cen_val'][i]

#             x1, y1  = t[h]['x_min'][i]  # extract the x and y coordinates from the array
#             x2, y2 = t[h]['x_max'][i]

#             # define the coordinates of the four corners of the rectangle
#             corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

#             # plot the lines connecting the corners of the rectangle
#             plt.plot([corner[0] for corner in corners], [corner[1] for corner in corners], '-k', linewidth=2)

#     directory = save_dir
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)
#     print("Plot saved")


# def draw_2d_partition_ridge_tree_sequool(t,settings,save_dir):

#     W = np.array([[1.5],[1],[-0.6]])
    
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t[h]['x'])): # node in a depth h

#             x, y = t[h]['x'][i]  # extract the x and y coordinates from the array
#             plt.plot(x, y, 'o', markersize=2, color='b')

#             origin = t[h]['x_min'][i]
# #            plt.plot([origin[0], origin[0] - W[1]], [origin[1], origin[1] + W[0]], color='red', linewidth=20)
#             plt.quiver(*origin, -W[1], W[0], color=['red'], scale=2)
#             plt.quiver(*origin, W[1], -W[0], color=['red'], scale=2)

#             origin = t[h]['x_max'][i]
#             plt.quiver(*origin, -W[1], W[0], color=['red'], scale=2,linewidth = h+1)
#             plt.quiver(*origin, W[1], -W[0], color=['red'], scale=2,linewidth = h+1)



#     directory = save_dir
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)
#     print("Plot saved")


# def two_ridge_tree_sequool(t1,t2,settings):

#     # W = np.array([[1.5],[1],[-0.6]])
#     W1 = np.array([[1.2,10],[1,1],[-0.6,-0.8]])
    
#     W = W1[:,0]
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t1[h]['x'])): # node in a depth h

#             x, y = t1[h]['x'][i]  # extract the x and y coordinates from the array
#             plt.plot(x, y, 'o', markersize=2, color='b')

#             origin = t1[h]['x_min'][i]
# #            plt.plot([origin[0], origin[0] - W[1]], [origin[1], origin[1] + W[0]], color='red', linewidth=20)
#             plt.quiver(*origin, -W[1], W[0], color=['red'], scale=2)
#             plt.quiver(*origin, W[1], -W[0], color=['red'], scale=2)

#             origin = t1[h]['x_max'][i]
#             plt.quiver(*origin, -W[1], W[0], color=['red'], scale=2)
#             plt.quiver(*origin, W[1], -W[0], color=['red'], scale=2)

#     W = W1[:,1]
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t2[h]['x'])): # node in a depth h

#             x, y = t2[h]['x'][i]  # extract the x and y coordinates from the array
#             plt.plot(x, y, 'o', markersize=2, color='b')

#             origin = t2[h]['x_min'][i]
# #            plt.plot([origin[0], origin[0] - W[1]], [origin[1], origin[1] + W[0]], color='red', linewidth=20)
#             plt.quiver(*origin, -W[1], W[0], color=['green'], scale=2)
#             plt.quiver(*origin, W[1], -W[0], color=['green'], scale=2)

#             origin = t2[h]['x_max'][i]
#             plt.quiver(*origin, -W[1], W[0], color=['green'], scale=2)
#             plt.quiver(*origin, W[1], -W[0], color=['green'], scale=21)

#     directory = "./2d_vis/"
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)
#     print("Plot saved")


# def two_ridge_tree_sequool_2(t1,t2,settings):

#     # W = np.array([[1.5],[1],[-0.6]])
#     W1 = np.array([[1.2,10],[1,1],[-0.6,-0.8]])

#     #W1 = np.array([[1,0],[1,1],[-0.9,-0.2]])
    
#     W = W1[:,0]
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t1[h]['x'])): # node in a depth h

#             x, y = t1[h]['x'][i]  # extract the x and y coordinates from the array
#             plt.plot(x, y, 'o', markersize=2, color='b')

#             origin = t1[h]['x_min'][i]
# #            plt.plot([origin[0], origin[0] - W[1]], [origin[1], origin[1] + W[0]], color='red', linewidth=20)
#             plt.quiver(*origin, W[0], W[1], color=['red'], scale=1)
#             plt.quiver(*origin, -W[0], -W[1], color=['red'], scale=1)

#             origin = t1[h]['x_max'][i]
#             plt.quiver(*origin, W[0], W[1], color=['red'], scale=1)
#             plt.quiver(*origin, -W[0], -W[1], color=['red'], scale=1)

#     W = W1[:,1]
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t2[h]['x'])): # node in a depth h

#             x, y = t2[h]['x'][i]  # extract the x and y coordinates from the array
#             plt.plot(x, y, 'o', markersize=2, color='b')

#             origin = t2[h]['x_min'][i]
# #            plt.plot([origin[0], origin[0] - W[1]], [origin[1], origin[1] + W[0]], color='red', linewidth=20)
#             plt.quiver(*origin, W[0], W[1], color=['green'], scale=1)
#             plt.quiver(*origin, -W[0], -W[1], color=['green'], scale=1)

#             origin = t2[h]['x_max'][i]
#             plt.quiver(*origin, W[0], W[1], color=['green'], scale=1)
#             plt.quiver(*origin, -W[0], -W[1], color=['green'], scale=1)

#     directory = "./2d_vis/"
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)
#     print("Plot saved")


# def draw_2d_tree_sequool_arbitary_direction(t,R,settings,save_dir):
    
#     for h in range(settings['h_max']+2):  # tree depth
#         for i in range(len(t[h]['x'])): # node in a depth h

# #            x,y = t[h]['x'][i]

#             x, y = rotation_to_original(R.T,t[h]['x'][i])[0]  # extract the x and y coordinates from the array
#             if point_in_unit_square(np.array([x,y])):
#                 plt.plot(x, y, 'o', markersize=2, color='b')

#     directory =save_dir
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)
#     print("Plot saved")


# def draw_2d_tree_sequool_two_arbitary_direction(t1,R1,t2,R2,settings,save_dir):

#     for h in range(settings['h_max']+2):  # tree depth
#         for i in range(len(t1[h]['x'])): # node in a depth h

#             x, y = rotation_to_original(R1.T,t1[h]['x'][i])[0]  # extract the x and y coordinates from the array
#             if point_in_unit_square(np.array([x,y])):
#                 plt.plot(x, y, 'o', markersize=2, color='b')

#     for h in range(settings['h_max']+2):  # tree depth
#         for i in range(len(t2[h]['x'])): # node in a depth h

#             x, y = rotation_to_original(R2.T,t2[h]['x'][i])[0]  # extract the x and y coordinates from the array
#             if point_in_unit_square(np.array([x,y])):
#                 plt.plot(x, y, 'o', markersize=2, color='r')


#     directory = save_dir
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     plt.savefig(filepath, dpi=300)
#     print("Plot saved")

# def draw_2d_partition_two_ridge_3(t,settings,save_dir,points,save= True):

    
#     for h in range(settings['h_max']):  # tree depth
#         for i in range(len(t[h]['x'])): # node in a depth h

#             x, y = t[h]['x'][i]  # extract the x and y coordinates from the array

#             if t[h]['cen_val'][i] != float('-inf'):
#                 plt.plot(x, y, 'o', markersize=1, color='b')
#             else:
#                 plt.plot(x, y, 'o', markersize=1, color='r')


#            #     plt.plot(t[h]['x'][i], t[h]['values'][i][1], 'og', markersize=5, markerfacecolor='green')
            


#             # x1, y1  = t[h]['x_min'][i]  # extract the x and y coordinates from the array
#             # x2, y2 = t[h]['x_max'][i]

#             # # define the coordinates of the four corners of the rectangle
#             # corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

#             # # plot the lines connecting the corners of the rectangle
#             # plt.plot([corner[0] for corner in corners], [corner[1] for corner in corners], '-k', linewidth=2)

#     # plotting the boundary as well
#     x = [p[0] for p in points]
#     x.append(points[0][0])
#     y = [p[1] for p in points]
#     y.append(points[0][1])
#     plt.plot(x, y, 'o-')

#     directory = save_dir
#     count = len([f for f in os.listdir(directory) if f.startswith("tree_")]) + 1
#     # Create the filename using the count and the basename
#     filename = f'tree_{count}.png'
#     filepath = os.path.join(directory, filename)

#     if save:
#         plt.savefig(filepath, dpi=300)
#         print("Plot saved")
   



    

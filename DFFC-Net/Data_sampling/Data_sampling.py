import os
import struct
import numpy as np


def data_process_direct(input_fts_dir, final_temp_dir, t1=6, t2=6):
    def trans_floor(x):
        t = int(x / 256)
        return int(x - t * 256)

    def read_fts(file_name):
        with open(file_name, "rb") as f:

            f.seek(267)
            s1 = struct.unpack('B', f.read(1))[0]
            f.seek(268)
            s2 = struct.unpack('B', f.read(1))[0]
            f.seek(269)
            s3 = struct.unpack('B', f.read(1))[0]
            f.seek(347)
            s4 = struct.unpack('B', f.read(1))[0]
            f.seek(348)
            s5 = struct.unpack('B', f.read(1))[0]
            f.seek(349)
            s6 = struct.unpack('B', f.read(1))[0]

            imgW = (s1 - 48) * 100 + (s2 - 48) * 10 + (s3 - 48)
            imgH = (s4 - 48) * 100 + (s5 - 48) * 10 + (s6 - 48)

            # Read image data
            allLength = os.path.getsize(file_name)
            dataSize = allLength - 2880
            f.seek(2880)
            f2 = f.read(dataSize)
            imgData = np.frombuffer(f2, dtype="ushort")
            pixelSize = len(imgData)
            frames = int(pixelSize / imgW / imgH)
            framePixels = imgH * imgW


            img = []
            for frame in range(frames):
                for pixel in range(framePixels):
                    index = frame * framePixels + pixel
                    t1_val = trans_floor(int(imgData[index] / 256.0))
                    t2_val = imgData[index] - t1_val * 256
                    v = int((t2_val * 256 + t1_val - 32768))
                    img.append(v)

            img = np.array(img, dtype='ushort').reshape([frames, imgH, imgW])
            return img, imgW, imgH, frames

    def save_image_frame(images, imgH, imgW, frames):

        image_frames = []
        for i in range(imgH):
            for j in range(imgW):
                for z in range(frames):
                    image_frames.append(images[z, i, j])
        return np.array(image_frames).reshape([imgH, imgW, frames])


    def sum_average(b, t, arr, imgH, imgW):
        sum_img = np.zeros([imgH, imgW])
        for i in range(b, b + t):
            sum_img += arr[:, :, i]
        return sum_img / t

    def modify_img(images, imgH, imgW, frames, t1, t2):
        back_image = sum_average(0, t2, images, imgH, imgW)
        imagenew = np.zeros([imgH, imgW, frames - t1], dtype=float)
        for i in range(frames - t1):
            image_average = sum_average(i, t1, images, imgH, imgW)
            imagenew[:, :, i] = image_average - back_image
        return imagenew



    for filedir in os.listdir(input_fts_dir):
        fts_subdir = os.path.join(input_fts_dir, filedir)

        temp_subdir = os.path.join(final_temp_dir, filedir)
        os.makedirs(temp_subdir, exist_ok=True)


        for filename in os.listdir(fts_subdir):

            if not filename.endswith('.FTS'):
                continue
            print(filename)
            # Read FTS data and perform conversion
            fts_path = os.path.join(fts_subdir, filename)
            img, imgW, imgH, frames = read_fts(fts_path)
            img_frame = save_image_frame(img, imgH, imgW, frames)


            images_modify = modify_img(img_frame, imgH, imgW, frames, t1, t2)

            temp_data = images_modify / 200
            temp_filename = f"{filename[:-4]}.npy"
            temp_save_path = os.path.join(temp_subdir, temp_filename)
            np.save(temp_save_path, temp_data)

            print(f"Processed and saved: {temp_save_path}")

    print("The entire process has been completed")



if __name__ == "__main__":

    input_fts_dir = r"../../orginal_data    /"  # Original FTS file root directory
    final_temp_dir = r"../../data_BP/"  # Final Temperature Data Save Directory

    data_process_direct(
        input_fts_dir=input_fts_dir,
        final_temp_dir=final_temp_dir,
        t1=6,  # Consecutive frame count parameter
        t2=6  # Background Frame Rate Parameter
    )
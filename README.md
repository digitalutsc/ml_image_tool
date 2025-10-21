# ❓ What is this

This repository includes a workflow that addresses common problems digital libraries encounter when archiving **images of corpora**. The common problem types are:
1. Rotation correction in multiples of 90 degrees.
2. Cropping the images so that a small fraction, or none, of the background remains.
3. Safely splitting images into two, in case the image set contains two-pagers.

The code is not wrapped in a UI yet, but the functionalities are all present (of course, future updates will be expected), even including QA and evaluation methods.
This workflow uses ⚙️methods⚙️ such as
1. Pre-processing using the EAST AI model,
2. CNN categorization,
3. Radon transform,
4. Fourier transform,
5. adaptive binarization,
6. opening & closing,
7. edge detection.

# 📄 The Image correction workflow

The image below illustrates the **five-step workflow** used in this project, along with their corresponding **visual representations** at each stage.

Each Python script in this repository is named according to the **step it implements**, making it easy to follow the full pipeline from start to finish.

Each script handles a distinct part of the image processing pipeline, from data loading and preprocessing to rotation correction and result export.

# 🤔 Why not merge all Python scripts into one automatic script?

Some steps make mistakes, so human intervention, like QA steps, needs to be done periodically. This is why step 4 has two branches, as shown in the
picture below. One of them is used to split the image, while the other one is used to merge the images that have been processed incorrectly.

# ❗ More about splitting pages

The image below shows two types of processing scripts for splitting and merging images, respectively. 

"Automatic" scripts are configured in a way so that the user only needs to drag the image in question into the processing folder, and the script will handle them one by one. You may want to use this type
of script because the computer can process the image concurrently with the user when the user is selecting images to be processed. Yes, selecting the two-
paged images are manual, and I haven't found a perfect automatic way to identify them. 

"Manual" scripts are to be used when you already know or have constructed a folder with two-page images only. Read the script comments for more information.

**You will also see** a script named "4 add_left_or_right_margin.py". This is used to add the left vertical portion of the right image to the left image and vice versa.
You may want to use this whenever the texts are too close to the central crease, or when the splitting lines are too close to the texts. After applying it, the reader
can be more confident that the cut did not accidentally split texts on the pages.

**What do you do after identifying and merging the poorly cut pages?** This online batch cropping tool is extremely fast and useful for this type of manual workflow. https://www.imgtools.co/crop-image
Remember to duplicate the images before uploading and splitting them on this website so that you can cut the left and right versions of the page at once.

---

## 📂 File Structure



![Flowchart (1)](https://github.com/user-attachments/assets/02650206-fd0c-48c2-9dde-4d84708d26ae)

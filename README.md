# 📄 Image Orientation Correction Workflow

The image below illustrates the **five-step workflow** used in this project, along with their corresponding **visual representations** at each stage.

Each Python script in this repository is named according to the **step it implements**, making it easy to follow the full pipeline from start to finish.

Each script handles a distinct part of the image processing pipeline, from data loading and preprocessing to rotation correction and result export.

# 🤔 Why not merge all Python scripts into one automatic script?

Some steps make mistakes, so human intervention, like QA steps, needs to be done periodically. In fact, this is why step 4 has its two branches in the
picture below. One of them is used to split the image, while the other one is used to merge the images split incorrectly.

# ❗ More about splitting pages

The image below shows two types of processing scripts for splitting and merging images, respectively. 

"Automatic" scripts are configured in a way so that the user only needs to drag the image in question into the processing_folder, and the script will handle them one by one. You may want to use this type
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



![Flowchart](https://github.com/user-attachments/assets/4f002ca8-4daa-4b3d-8761-bbb7624b2e15)

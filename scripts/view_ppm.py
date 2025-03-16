import sys
import matplotlib.pyplot as plt
import imageio.v3 as iio

def view_ppm(file_path):
    try:
        # Read the PPM file
        image = iio.imread(file_path)
        
        # Display the image
        plt.imshow(image)
        plt.axis("off")  # Hide axes
        plt.title(file_path)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_ppm.py <filename.ppm>")
        sys.exit(1)
    
    ppm_file = sys.argv[1]
    view_ppm(ppm_file)

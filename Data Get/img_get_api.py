#########################   Instruction   ##########################
####Used to capture images from the Internet through official API###
####################################################################

import os
import requests

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")

def scrape_unsplash_images(api_key, query, num_images, save_dir):
    create_directory(save_dir)

    base_url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    page = 1
    downloaded_count = 0

    while downloaded_count < num_images:
        params = {
            "query": query,
            "page": page,
            "per_page": 30
        }

        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            for result in data.get("results", []):
                if downloaded_count >= num_images:
                    break

                img_url = result["urls"]["small"]
                img_extension = ".jpg"
                save_path = os.path.join(save_dir, f"image_{downloaded_count + 1}{img_extension}")

                download_image(img_url, save_path)
                downloaded_count += 1

            if len(data.get("results", [])) == 0:
                print("No more images available.")
                break

            page += 1
        except requests.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            break

    print(f"Finished downloading {downloaded_count} images.")

if __name__ == "__main__":
    api_key = "VJtw8EvuBTZiKBOa5V4bKmWLnE9019cZrt9jTLX78I4"
    scrape_unsplash_images(api_key, "cat", 1000, "../data/cat")
    scrape_unsplash_images(api_key, "dog", 1000, "../data/dog")
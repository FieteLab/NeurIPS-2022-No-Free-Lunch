import gdown
import os
import zipfile


project_urls = {
    'banino': 'https://drive.google.com/file/d/1K5TnYgqhD6YuYFhK-OIGduMe5a-D2-x-/view?usp=sharing',
    'sorscher': 'https://drive.google.com/file/d/1dks4CTG3T65FxxXM59UfNt_ma3BSyo4e/view?usp=sharing',
    'whittington': 'https://drive.google.com/file/d/1WZpgm8PXUCb5iG594btLqc3nEZj_Ij_T/view?usp=sharing',
    'nayebi': 'https://drive.google.com/file/d/1Sfalj6RLH9bYCpzytFp5USHo1Mw4ilF0/view?usp=sharing',
}


def download_zips(zips_dir_path: str = 'zips'):

    os.makedirs(zips_dir_path, exist_ok=True)

    for project, url in project_urls.items():
        path_to_zipfile = os.path.join(zips_dir_path, f'{project}.zip')
        gdown.download(url=url, output=path_to_zipfile, quiet=False, fuzzy=True)
        print(f'Downloaded {project} from {url}')


def unzip_zips(zips_dir_path: str = 'zips',
               project_code_dir: str = 'investigations/subprojects'):

    os.makedirs(project_code_dir, exist_ok=True)

    for project in project_urls.keys():
        path_to_zipfile = os.path.join(zips_dir_path, f'{project}.zip')
        path_to_write_to = os.path.join(project_code_dir, project)
        with zipfile.ZipFile(path_to_zipfile, 'r') as zip_fp:
            zip_fp.extractall(path_to_write_to)


if __name__ == '__main__':
    download_zips()
    unzip_zips()

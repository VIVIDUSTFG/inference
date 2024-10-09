import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Run a specified AI model script.')
    parser.add_argument('--model-name', type=str,
                        help='The name of the model to run')
    parser.add_argument('--modality', type=str,
                        help='The modality of the model to run')
    parser.add_argument('--data-path', type=str,
                        help='The path to the folder containing the data')
    parser.add_argument('--models-path', type=str,
                        help='The path to the folder containing the models')

    args = parser.parse_args()

    model_name = args.model_name
    modality = args.modality
    data_path = args.data_path
    models_path = args.models_path

    rgb_list_file_path = os.path.join(data_path, "rgb.list")
    audio_list_file_path = os.path.join(data_path, "audio.list")

    rgb_npy_files = []
    audio_npy_files = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('rgb.npy'):
                absolute_path = os.path.join(root, file)
                rgb_npy_files.append(absolute_path)
            if modality == 'rgb_and_audio' and file.endswith('vggish.npy'):
                absolute_path = os.path.join(root, file)
                audio_npy_files.append(absolute_path)

    with open(rgb_list_file_path, 'w') as list_file:
        for path in rgb_npy_files:
            list_file.write(path + '\n')

    if modality == 'rgb_and_audio':
        with open(audio_list_file_path, 'w') as list_file:
            for path in audio_npy_files:
                list_file.write(path + '\n')

    model_script_path = os.path.join(models_path, model_name)
    if not os.path.exists(model_script_path):
        print(f"Model script not found: {model_script_path}")
        return

    command = ["python", "infer.py",
               "--rgb-list", rgb_list_file_path, "--output-path", data_path]
    print(f"Running model: {model_name} with modality: {modality}")
    if modality == 'rgb_and_audio':
        print(f"Audio list file: {audio_list_file_path}")
        command.extend(["--audio-list", audio_list_file_path])

    subprocess.run(command, check=True, cwd=model_script_path)


if __name__ == '__main__':
    main()

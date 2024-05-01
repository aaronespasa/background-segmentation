import os
import shutil

# Definir el directorio base y los subdirectorios
base_dir = 'data'
src_images = os.path.join(base_dir, 'img')
src_masks = os.path.join(base_dir, 'masks_machine')

dst_dirs = {
    'training': ('images/training', 'annotations/training'),
    'validation': ('images/validation', 'annotations/validation'),
    'test': ('images/test', 'annotations/test')
}

# Función para mover las imágenes y máscaras a los directorios correctos
def move_files(start_index, end_index, folder_type):
    for i in range(start_index, end_index + 1):
        image_name = f"{i}.jpg"
        mask_name = f"{i}.png"
        
        # Construir rutas de origen
        src_image_path = os.path.join(src_images, image_name)
        src_mask_path = os.path.join(src_masks, mask_name)
        
        # Construir rutas de destino
        dst_image_path = os.path.join(base_dir, dst_dirs[folder_type][0], image_name)
        dst_mask_path = os.path.join(base_dir, dst_dirs[folder_type][1], mask_name)
        
        # Mover archivos
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_mask_path, dst_mask_path)

if __name__ == "__main__":
    # Crear los directorios destino si no existen
    for key in dst_dirs:
        os.makedirs(os.path.join(base_dir, dst_dirs[key][0]), exist_ok=True)
        os.makedirs(os.path.join(base_dir, dst_dirs[key][1]), exist_ok=True)

    # Números de imágenes para cada conjunto
    num_training = 478
    num_validation = 80
    num_test = 31

    # Dividir los archivos y moverlos a sus respectivas carpetas
    move_files(1, num_training, 'training')
    move_files(num_training + 1, num_training + num_validation, 'validation')
    move_files(num_training + num_validation + 1, num_training + num_validation + num_test, 'test')

    print("Los archivos han sido distribuidos correctamente.")

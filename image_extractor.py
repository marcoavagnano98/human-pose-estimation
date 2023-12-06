import os
import cv2
import numpy as np
import pyrealsense2 as rs
#import matplotlib.pyplot as plt
#from IPython.display import display, clear_output
#import xml.etree.ElementTree as ET

### Funzione per estrarre immagini (rgb, depth 16bit, depth 8bit, infrared) da file .bag
# bag_filename vuole una stringa col percorso al file .bag
# es. "./Video_Technogym/Acquisizioni_Raw/20230908_160009.bag"
def extract_frames_from_bag(bag_filename):

    # Estraggo il nome del file senza percorso e estensione
    bag_name = os.path.splitext(os.path.basename(bag_filename))[0]

    # Creazione delle cartelle se non esistono
    unaligned_color_folder = f"./frames/{bag_name}/rgb_unaligned/"
    aligned_color_folder = f"./frames/{bag_name}/rgb_aligned/"
    depth_16bit_folder = f"./frames/{bag_name}/depth_16bit/"
    depth_8bit_folder = f"./frames/{bag_name}/depth_8bit/"
    infrared_folder = f"./frames/{bag_name}/infrared/"

    for folder in [unaligned_color_folder, aligned_color_folder, depth_16bit_folder,depth_8bit_folder,infrared_folder]:
        os.makedirs(folder, exist_ok=True)

    # Configurazione della pipeline per il file BAG (playback.set_real_time(False) è essenziale per non perdere frame)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_filename)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Allinea il frame rgb (più piccolo) al frame depth (uguale al frame ir, entrambi più grandi)
    align = rs.align(rs.stream.depth)

    # Dichiarazione di variabili per tenere traccia del conteggio dei frame
    frame_count = 0
    frameNum_prev = 0

    try:
        while True:
            frames = pipeline.wait_for_frames() 
            frameNum = frames.get_frame_number()

            aligned_frames = align.process(frames)

            unaligned_rgb_frame = frames.get_color_frame()

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            infrared_frame = aligned_frames.get_infrared_frame()

            # Ho notato che l'ultimo frame è presente due volte nell'immagine rgb;
            # da qui verifico se ci sono due frame uguali successivi e se si esco al loop
            print(frameNum, frameNum_prev)


            # Salvo il frame rgb non allineato (senza bordi neri)
            if(unaligned_rgb_frame):
                unaligned_color_image = np.asanyarray(unaligned_rgb_frame.get_data())
                unaligned_color_frame_filename = os.path.join(
                    unaligned_color_folder, f"{frame_count:06d}.jpg"
                )
                cv2.imwrite(unaligned_color_frame_filename, cv2.cvtColor(unaligned_color_image, cv2.COLOR_BGR2RGB))

            # Se presenti, salva il frame a colori come immagine 000001.jpg
            if(color_frame):
                color_image = np.asanyarray(color_frame.get_data())
                color_frame_filename = os.path.join(
                    aligned_color_folder, f"{frame_count:06d}.jpg"
                )
                cv2.imwrite(color_frame_filename, cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            # Se presente, salva frame di profondità a 16 ed 8 bit
            if(depth_frame):
                # Salva il frame di profondità a 16bit come immagine 000001.png
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_frame_filename = os.path.join(
                    depth_16bit_folder, f"{frame_count:06d}.png"
                )
                cv2.imwrite(depth_frame_filename, depth_image)
            
                # Applica una colormap sull'immagine a colori (l'immagine va prima convertita a 8-bit per pixel)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                #Il seguente codice commentato serve per mostrare a video l'immagine color e depth sovrapposte:
                #merged_image = cv2.addWeighted(color_image, 0.5, depth_colormap, 0.5, 0)

                # Salva il frame di profondità ad 8bit come immagine 000001.png (per visualizzazione)
                depth_frame_filename = os.path.join(
                    depth_8bit_folder, f"{frame_count:06d}.png"
                )
                cv2.imwrite(depth_frame_filename, depth_colormap)


            # Se presente, salva il frame infrarosso come immagine 000001.jpg
            if(infrared_frame):
                infrared_image = np.asanyarray(infrared_frame.get_data())
            
                infrared_frame_filename = os.path.join(
                    infrared_folder, f"{frame_count:06d}.jpg"
                )
                cv2.imwrite(infrared_frame_filename, infrared_image)

                ir_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(infrared_image, alpha=0.03), cv2.COLORMAP_JET
                )
                #Il seguente codice commentato serve per mostrare a video l'immagine color e ir sovrapposte:
                #merged_image = cv2.addWeighted(color_image, 0.5, ir_colormap, 0.5, 0)

            # Incrementa il conteggio dei frame
            frame_count += 1    

            # Da decommentare per visualizzare immagini a schermo:

            # Ruota l'immagine in senso orario di 90 gradi 
            #rotated_image = cv2.rotate(merged_image, cv2.ROTATE_90_CLOCKWISE)

            # Visualizza l'immagine
            #cv2.imshow("Video", rotated_image)

            # Interrompi il loop quando viene premuto 'q'
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #   break
            if frameNum < frameNum_prev:
                break
            else:
                frameNum_prev = frameNum

    finally:
        # Chiudi la pipeline e il video
        pipeline.stop()
        cv2.destroyAllWindows()

def extract_bag_from_folder(folder_path):
    bag_paths = os.listdir(folder_path)
    for bag in bag_paths:
        filename = bag.split('.')[0]
        if filename not in os.listdir("frames"):
            print(f"Extracting {filename} ....")
            extract_frames_from_bag(os.path.join(folder_path, bag))


if __name__ == "__main__":
    #extract_frame_from_("/media/marco/78062BEF062BAD56/Users/Marco/Documents/lontano")
   # extract_bag_from_folder("/media/marco/78062BEF062BAD56/Users/Marco/Documents/vicino")
    extract_frames_from_bag("/media/marco/78062BEF062BAD56/Users/Marco/Documents/lontano/20231201_100731.bag")







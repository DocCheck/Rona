# add python path inside ROS2 workspace
import sys
sys.path.insert(0, "src/rona_pkg/rona_pkg")

from rclpy.node import Node

from custom_interfaces.srv import VoiceSrv
from custom_interfaces.msg import ExitMsg
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from openwakeword.model import Model
from .robot_handler import Rona
from playsound import playsound
from queue import Queue
from time import sleep
from sty import fg

import speech_recognition as sr
import numpy as np
from rona_pkg import config_rona
import whisper
import pyaudio
import torch
import rclpy
import time
import io

### fg(4) -> dark blue log ###


class VoiceRecognitionService(Node):

    # Init
    def __init__(self):
        #################################
        ############## ROS ##############
        #################################

        # Initialize node
        super().__init__('voice_recognition_service')
        self.srv = self.create_service(VoiceSrv, 'voice_recognition_service', self.service_callback)
        self.arm = Rona()

        # Speaker single call variable
        self.system_started = False        

        # Exit subscriber
        self.create_subscription(ExitMsg, 'exit_message', self.exit_callback, 1)

        #################################
        ####### Voice recognition #######
        #################################

        ### OPENWAKEWORD ###
        ww_model_path = config_rona.path_ww_model
        inference_framework = 'onnx'
        self.time_awake = config_rona.time_awake
        
        # Get microphone stream
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1280
        self.pyaudio = pyaudio.PyAudio()

        self.owwModel = Model(wakeword_models=[ww_model_path], inference_framework=inference_framework)
        n_models = len(self.owwModel.models.keys())
        
        ### WHISPER ###
        self.whisper_model = config_rona.whisper_model
        self.arg_non_english = False

        mic_name = 'pulse'

        if not mic_name or mic_name == 'list':
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    self.source = sr.Microphone(sample_rate=self.RATE, device_index=index)
                    break

        # The last time a recording was retrieved from the queue.
        self.phrase_time = None
        # Current raw audio bytes.
        self.last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 1000
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False

        # Load / Download model
        model = self.whisper_model
        if self.whisper_model != "large" and not self.arg_non_english:
            model = model + ".en"
        self.audio_model = whisper.load_model(model)
        print("Model loaded.\n")

        self.record_timeout = 2.0
        self.phrase_timeout = 3.0

        self.temp_file = NamedTemporaryFile().name
        self.transcription = ['']

        # Command list words
        self._syntax_dict_commands = {
            "cancel": ["cancel", "cancer"],
            "change": ["change", "chain", "strange", "range"],
            "give": ["give", "gift", "live"],
            "recalibration": ["recalibration", "recalibrate"],
            "reset": ["reset", "set", ],
            "return": ["return", "turn", "burn", "refund"],
            "save": ["save", "wave", "safe", ],
            "shut down": ["shut down", "shutdown", "down", "shut"],
            "take": ["take", "make", "cake", "stake"]
        }

        self._single_commands = ["recalibration", "reset", "return", "save", "shut down"]

        self._syntax_dict_instrument = {
            "handle": ["handle", "hand", "candle", "hand-o-number", "hand-down", "hand-on"],
            "needle": ["needle", "need", "dull", "knee"],
            "scalpel": ["scalpel", "couple", "carbon", "scalper", "scarple", "kowlpool"],
            "scissors": ["scissors"],
            "tweezers": ["tweezers", "teasers", "tweet"]
        }

        self._syntax_dict_adjectives = {
            "three": ["three", "3", "tree"],
            "four": ["four", "4", "core", "flour", "floor"],
            "clenched": ["clenched", "blend", "cleanse", "cleansed"],
            "narrow": ["narrow", "narc", "marrow", "arrow"],
            "blunt": ["blunt", "blood", "bunt"],
            "pointed": ["pointed", "point"],
            "slim": ["slim", "slum"],
            "standard": ["standard", "dart"],
            "surgical": ["surgical"]
        }

        self._special_instruments = {
            "tweezers": ["standard", "slim", "surgical"],
            "handle": ["three", "four"],
            "scalpel": ["clenched", "narrow"],
            "scissors": ["pointed", "blunt"]
        }
        self.full_path = config_rona.path_audio_files
        self.final_command = ""
        self.final_instrument = ""

    # Method for dictionary search
    def dict_search(self, words, input_dict):
        result = None
        keys = input_dict.keys()
        for word in words:
            for key in keys:
                if word in input_dict[key]:
                    result = key
                    print(f'The best match is {result}!')
                    return result
                
        return result

    # Method for intent recognition
    def intent_recognizer(self, line):
        words = line.split()
        command_result = self.dict_search(words, self._syntax_dict_commands) ### Command 
        instrument_result = self.dict_search(words, self._syntax_dict_instrument) ### Instrument
        adj_result = self.dict_search(words, self._syntax_dict_adjectives) ### Adjective

        self.get_logger().info(f'Command: {command_result}; Instrument: {instrument_result}; Adjective: {adj_result}')
        return command_result, instrument_result, adj_result

    # Raw sound input callback
    def record_callback(self, _, audio: sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            self.data_queue.put(data)
    
    ### Wake-word detection ###
    def wakeword_detection(self):
        
        # Init/Reset parameters
        self.awake = False
        self.owwModel.reset()
        # Open audio stream
        mic_stream = self.pyaudio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
                
        # Until system is awake
        while not self.awake:
            # Get audio
            audio = np.frombuffer(mic_stream.read(self.CHUNK), dtype=np.int16)

            # Feed to openwakeword model
            prediction = self.owwModel.predict(audio)

            for mdl in self.owwModel.prediction_buffer.keys():
                # Add scores
                scores = list(self.owwModel.prediction_buffer[mdl])
                if scores[-1] > 0.8 and len(scores) > 10:
                    self.get_logger().warn("Wake-word detected!")
                    self.awake = True
                    mic_stream.stop_stream()
                    mic_stream.close()
                    # Create a background thread that will pass us raw audio bytes.

                    self.stopper = self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)
                    break

    ### Command detection ###
    def listening_for_command(self):

        self.arm.open_gripper() 
        start = time.time()
        
        # While system is awake
        while self.awake: 
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not self.data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                    self.last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                self.phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(self.temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = self.audio_model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    self.transcription.append(text)
                else:
                    self.transcription[-1] = text

                # Take last transcription only and clean
                line = self.transcription[-1]
                line = line.lower()
                line = line.strip('~`!@$%^&*()_-+={}[]|\:;<,>.?/')
                
                # Intent recognizer
                final_command, final_instrument, final_adjective = self.intent_recognizer(line)

                if final_command is not None and not self.fast_mode:
                    # Prepare message command variable
                    self.final_command = final_command
                    if final_command in self._single_commands:
                        self.get_logger().info(fg(4) + f"Match has been found. You have chosen {self.final_command}!" + fg.rs)
                        self.final_instrument = ""
                        self.awake = False
                        self.listening = False
                        break
                    
                    else:
                        if final_instrument is not None:
                            if final_instrument == "needle":
                                self.get_logger().info(fg(4) + f"Match has been found. You have chosen {self.final_command}!" + fg.rs)
                                self.get_logger().info(fg(4) + f"Instrument chosen is {final_instrument}!" + fg.rs)
                                
                                # Prepare message instrument variable
                                self.final_instrument = final_instrument
                                self.awake = False
                                self.listening = False
                                break

                            else:
                                if final_adjective is not None and final_adjective in self._special_instruments[final_instrument]:
                                    self.get_logger().info(fg(4) + f"Match has been found. You have chosen {self.final_command}!" + fg.rs)
                                    self.final_instrument = final_adjective + " " + final_instrument
                                    self.get_logger().info(fg(4) + f"Instrument chosen is {self.final_instrument}!" + fg.rs)
                                    self.awake = False
                                    self.listening = False
                                    break
                                else:
                                    print("Instrument invalid. Specify better what instrument you want to have.")
                                    
                                    # Prepare message instrument variable
                                    self.final_instrument = final_instrument
                                    
                                    # Reset time to give user more time to say his instrument and go to fast mode
                                    sound = self.full_path + f'choose_{self.final_instrument}_variation.mp3'
                                    playsound(sound, block=True)
                                    start = time.time()
                                    self.fast_mode = True

            elif self.fast_mode:
                if final_adjective is not None and final_adjective in self._special_instruments[self.final_instrument]:
                    self.get_logger().info(fg(4) + f"Match has been found. You have chosen {self.final_command}!" + fg.rs)
                    self.final_instrument = final_adjective + " " + self.final_instrument
                    self.get_logger().info(fg(4) + f"Instrument chosen is {self.final_instrument}!" + fg.rs)
                    self.awake = False
                    self.listening = False
                    self.fast_mode = False
                    break

            if time.time() - start > self.time_awake:#  X seconds listening after wake word
                self.get_logger().warn("Exiting command listener!")
                self.awake = False   
                self.fast_mode = False
            
        # Reset for wake word
        self.arm.close_gripper(wait=False)  

    # Clear audio input queue
    def clear_queue(self):
        # Clear Queue
        with self.data_queue.mutex:
            self.data_queue.queue.clear()

        while not self.data_queue.empty():
            try:
                self.data_queue.get(block=False)
            except:
                continue
            self.data_queue.task_done()

        # Flush stdout.
        print('', end='', flush=True)

    # Main voice recognition callback function
    def service_callback(self, request, response):
        
        # Initiate closed gripper
        self.arm.close_gripper() 
        
        ### WHISPER ###
        with self.source:
                self.recorder.adjust_for_ambient_noise(self.source)
                
        # Cue the user that we're ready to go.
        self.listening = True        
        self.fast_mode = False
        self.get_logger().info(fg(4) + "Voice detection has started. Say something..." + fg.rs)
     
        # Main loop
        while self.listening:
            
            try:       
                # Listening for wake word
                self.get_logger().info("Listening to wake word...")
                self.wakeword_detection()    
                                
                # Listening for command
                self.get_logger().info("Listening for command...")
                self.listening_for_command()

                # Background listener thread stopper
                self.stopper(wait_for_stop=True)

                # Clear queue
                self.clear_queue()

                # Infinite loops are bad for processors, must sleep.
                sleep(0.1)

            except KeyboardInterrupt:
                break

            except sr.UnknownValueError:
                self.get_logger().error(fg(4) + "Could not understand audio" + fg.rs)
                

            except sr.RequestError as e:
                self.get_logger().error(fg(4) + "Could not request results from Speech Recognition service; {0}".format(e) + fg.rs)


            except sr.WaitTimeoutError as e:
                self.get_logger().error(fg(4) + "Service timed out because of too long silence; {0}".format(e) + fg.rs)
        
        # Clear queue
        self.clear_queue()

        # Prepare service response
        response.command = self.final_command
        response.instrument = self.final_instrument
        self.get_logger().info(fg(4) + self.final_command + fg.rs)
        return response

    # Exit callback
    def exit_callback(self, msg):
        if msg.exit == True:
            exit()


def main():
    # Start voice recognition service and spin node
    rclpy.init(args=None)
    voice_recognition_service = VoiceRecognitionService()
    voice_recognition_service.get_logger().info(fg(4) + "Voice recognition service has been started" + fg.rs)
    rclpy.spin(voice_recognition_service)

    voice_recognition_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
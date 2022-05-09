import os, sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo
import numpy as np


frame_format = 'BGR'

Gst.init()

width, height = 600, 600
pipeline = Gst.parse_launch(f'''
    rtsp://admin:Abcd1234@113.190.240.99:554/Streaming/Channels/1 !
    decodebin !
    videoconvert !
    video/x-raw,format={frame_format} !
    fakesink name=s  
''')

def on_frame_probe(pad, info):
    buf = info.get_buffer()
    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    caps_format = pad.get_current_caps().get_structure(0)
    height, width = caps_format.get_value('height'), caps_format.get_value('width')
    video_format = GstVideo.VideoFormat.from_string(
        caps_format.get_value('format'))
    print(f'video format: {video_format}')
    # print(f'{width*height*3} / {len(map_info.data)}')
    image_array = buffer_to_image_array(buf, pad.get_current_caps())
    print(image_array.shape)

    #print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    return Gst.PadProbeReturn.OK


def buffer_to_image_array(buf, caps):
    pixel_bytes = 3
    caps_structure = caps.get_structure(0)
    height, width = caps_structure.get_value('height'), caps_structure.get_value('width')

    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if is_mapped:
        try:
            image_array = np.ndarray(
                (height, width, pixel_bytes),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy() # extend array lifetime beyond subsequent unmap
            #return preprocess(image_array[:,:,:3]) # RGBA -> RGB
            
            return image_array
        finally:
            buf.unmap(map_info)


pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

try:
    c=0
    while c < 30:
      c+1
      msg = pipeline.get_bus().timed_pop_filtered(
          Gst.SECOND,
          Gst.MessageType.EOS | Gst.MessageType.ERROR
      )
      if msg:
          text = msg.get_structure().to_string() if msg.get_structure() else ''
          msg_type = Gst.message_type_get_name(msg.type)
          print(f'{msg.src.name}: [{msg_type}] {text}')
          break
finally:
    
    pipeline.set_state(Gst.State.NULL)
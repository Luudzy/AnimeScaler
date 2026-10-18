import contextlib
from types import SimpleNamespace
import unittest
import numpy as np
from server import Renderer
from protocol import HEADER, MAGIC

class FrameReuseTests(unittest.TestCase):
    def renderer(self):
        r=Renderer.__new__(Renderer)
        r.torch=SimpleNamespace(inference_mode=lambda:contextlib.nullcontext())
        r.select_model=lambda name:None
        r.shape=(40,40,80,80);r.last_transport='rgb';r.reusable_frame=True;r.frame_revision=5
        pixels=np.full((40,40,4),100,dtype=np.uint8)
        r.host_input=SimpleNamespace(numpy=lambda:pixels)
        r.host_output=SimpleNamespace(numpy=lambda:np.full((80,80,3),120,dtype=np.uint8))
        return r

    def packet(self,frame_id=7):
        return HEADER.pack(MAGIC,frame_id,40,40,80,80)+bytes([100])*(40*40*4)

    def test_identical_frame_returns_compact_reply_with_new_id(self):
        r=self.renderer()
        body,_=r.render(self.packet(8),'adore-lite','rgb',allow_repeat=True)
        self.assertEqual(HEADER.unpack(body),(0x34435341,8,80,80,80,80))
        self.assertTrue(r.last_reused)
        self.assertFalse(r.last_warmup)
        self.assertEqual(r.frame_revision,5)

    def test_legacy_client_gets_full_pixels_even_when_inference_reused(self):
        r=self.renderer()
        body,_=r.render(self.packet(),'adore-lite','rgb')
        self.assertEqual(len(body),24+80*80*3)
        self.assertEqual(body[24:],bytes([120])*(80*80*3))

    def test_one_changed_byte_forces_inference_and_invalidates_failed_cache(self):
        r=self.renderer();data=bytearray(self.packet());data[-1]=99
        # No GPU buffers are installed: reaching inference must fail, rather than
        # reusing a frame which differs even at the final pixel.
        with self.assertRaises(AttributeError):r.render(data,'adore-lite','rgb',allow_repeat=True)
        self.assertFalse(r.reusable_frame)
        self.assertFalse(r.last_reused)

if __name__=='__main__':unittest.main()

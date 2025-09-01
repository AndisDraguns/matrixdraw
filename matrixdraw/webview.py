from io import BytesIO
import base64

from matrixdraw.draw import Matrix


def matrix_to_html(matrix: Matrix) -> str:
    buffer = BytesIO()
    matrix.save(buffer)
    buffer.seek(0)
    buffer_val = buffer.getvalue()
    if matrix.conf.raster:
        img_str = base64.b64encode(buffer_val).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_str}" />'
    else:
        return buffer_val.decode('utf-8')

// automatically generated by the FlatBuffers compiler, do not modify

import java.nio.*
import kotlin.math.sign
import com.google.flatbuffers.*

@Suppress("unused")
@ExperimentalUnsignedTypes
class Rapunzel : Struct() {

    fun __init(_i: Int, _bb: ByteBuffer)  {
        __reset(_i, _bb)
    }
    fun __assign(_i: Int, _bb: ByteBuffer) : Rapunzel {
        __init(_i, _bb)
        return this
    }
    val hairLength : Int get() = bb.getInt(bb_pos + 0)
    fun mutateHairLength(hairLength: Int) : ByteBuffer = bb.putInt(bb_pos + 0, hairLength)
    companion object {
        fun createRapunzel(builder: FlatBufferBuilder, hairLength: Int) : Int {
            builder.prep(4, 4)
            builder.putInt(hairLength)
            return builder.offset()
        }
    }
}
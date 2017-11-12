using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CharRNNCNTK
{
    class Stabilize
    {
        public static Function Build<TElementType>(Variable input, DeviceDescriptor device, string outputName = "Stabilize")
        {
            System.Diagnostics.Debug.Assert(typeof(TElementType) == typeof(float) || typeof(TElementType) == typeof(double));
            bool isFloatType = typeof(TElementType) == typeof(float);
            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new Parameter(new NDShape(), f.DataType, .99537863 /* 1/f*ln(e^f-1)*/, device, "stabilizerParameter")))),
                "beta");
            return CNTKLib.ElementTimes(beta, input, outputName);
        }

        public static Function Build(Variable input, DeviceDescriptor device, string outputName = "Stabilize")
        {
            return Build<float>(input, device, outputName);
        }
    }
}

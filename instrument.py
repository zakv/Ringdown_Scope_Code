import os
import time
import numpy
np=numpy

class usbtmc:
    """Simple implementation of a USBTMC device driver, in the style of visa.h"""

    def __init__(self, device):
        self.device = device
        try:
            self.FILE = os.open(device, os.O_RDWR)
        except OSError:
            raise OSError('Plug in scope and run ' +
                    '"sudo chmod a+rw /dev/usbtmc0"')

    def write(self, command):
        os.write(self.FILE, command);
        #time.sleep(.1)

    def read(self, length = 10000):
        return os.read(self.FILE, length)

    def getName(self):
        self.write("*IDN?")
        return self.read(300)

    def sendReset(self):
        self.write("*RST")


class AgilentScope:
    """Class to control a Rigol DS1000 series oscilloscope"""
    def __init__(self, device):
        self.meas = usbtmc(device)

        self.name = self.meas.getName()

        print self.name

    def write(self, command):
        """Send an arbitrary command directly to the scope"""
        self.meas.write(command)

    def read(self, length=10000):
        """Read an arbitrary amount of data directly from the scope"""
        return self.meas.read(length)

    def get_channel_data(self, channel_number):
        """Automatically returns the y-values in volts for the specified channel"""
        self.write(":WAV:POIN:MODE RAW")
        self.write(":WAV:FORM BYTE")
        self.write(":WAVEFORM:POINTS 10240")
        self.write(":STOP")
        self.write(":WAV:DATA? CHAN%d" % channel_number)
        #time.sleep(0.01)
        number_digits=self.read(2)
        number_digits=int(number_digits[1])
        number_data_points=int(self.read(number_digits))
        rawdata = self.read(number_data_points)
        data = numpy.frombuffer(rawdata, 'B')
        while len(data)<number_data_points:
            print  ("Data length should be %d, but is only %d" %
                    (number_data_points,len(data)))
            raise BufferError( "Data length should be %d, but is only %d" %
                (number_data_points,len(data)) )
        return data

    def check_channel_scaling(self,channel_number=1):
        """Gets first entry as an ASCII string and prints out conversion as well"""
        #Get and convert data usual way
        series=self.get_multiple_traces(n_traces=1,
                channel_number=channel_number)
        converted_data=series.converted_data[0,0]
        #Get data converted by scope to ascii string for comparison
        self.write("WAV:FORM ASCII")
        self.write(":WAV:DATA? CHAN%d" % channel_number)
        number_digits=self.read(2)
        number_digits=int(number_digits[1])
        number_data_points=int(self.read(number_digits))
        ascii_data= self.read(25).split(',')[0]
        #Print data for comparison
        print "ASCII: %s, and converted: %f" % (ascii_data, converted_data)
        return

    def get_single_trace(self,channel_number):
        """Takes a new trace and returns the channel_data"""
        self.write(":SINGLE") #Set trigger to single
        self.write(":TRIGger:STATus?") #Wait until scope takes data
        trigger_status=self.read()
        while trigger_status!='STOP\n':
            time.sleep(0.01)
            self.write(":TRIGger:STATus?")
            trigger_status=self.read()
        try:
            data=self.get_channel_data(channel_number) #Retrieve scope data
        except BufferError:
            print "Trying to take another trace..."
            data=self.get_single_trace(channel_number)
        return data

    def get_multiple_traces(self,n_traces=10,channel_number=1):
        """Takes multiple traces and returns an array with each trace as a row"""
        #Prepare object to hold data
        series=Measurement_Series(self)
        series.get_scope_settings()
        #Start taking data
        one_data=self.get_single_trace(channel_number)
        all_data=numpy.zeros([n_traces,len(one_data)])
        all_data[0]=one_data
        for j in range(1,n_traces):
            one_data=self.get_single_trace(channel_number)
            all_data[j]=one_data
        series.channel_data=all_data
        self.unlock()
        return series

    def unlock(self):
        """Unlocks the buttons on the scope for manual use"""
        self.write(":KEY:LOCK DISable")

    def reset(self):
        """Reset the instrument"""
        self.meas.sendReset()


class Measurement_Series(object):
    """Handles the data from one series of measurements"""

    def __init__(self,scope):
        """Initializes a Measurement_Series instance from an AgilentScope"""
        self.scope=scope
        self.channel_data=np.array([])
        self.time_offset=None
        self.delta_t=None
        self.y_reference=None
        self.y_origin=None
        self.y_increment=None

    def get_scope_settings(self):
        """Sets attributes to reflect current scope settings"""
        scope=self.scope
        scope.write(":WAV:XOR?")
        self.time_offset=float(scope.read(20))
        scope.write(":WAV:XINC?")
        self.delta_t=float(scope.read(20))
        scope.write(":WAV:YREF?")
        self.y_reference=float(scope.read(20))
        scope.write(":WAV:YOR?")
        self.y_origin=float(scope.read(20))
        scope.write(":WAV:YINC?")
        self.y_increment=float(scope.read(20))

#This wasn't working for some reason
    @property
    def converted_data(self):
        """Channel_data converted to real voltages (in volts)"""
        converted_data=((self.y_reference-1)-self.channel_data)*self.y_increment-self.y_origin
        return converted_data

    @property
    def time_data(self):
        """The times (in seconds) corresponding to the channel_data"""
        length=self.trace_length
        time_data=numpy.linspace(0.,length-1,length)
        time_data=time_data*self.delta_t+self.time_offset
        return time_data

    @property
    def n_traces(self):
        """The number of traces in this Measurement_Series instance"""
        return self.channel_data.shape[0]

    @property
    def trace_length(self):
        """The number of data points in each trace"""
        return self.channel_data.shape[1]

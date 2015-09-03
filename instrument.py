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
            raise OSError('Plug in scope and run "sudo chmod a+rw usbtmc0"')

        # TODO: Test that the file opened

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

    def convert_channel_data(self,data):
        """Converts arrays of unsigned 8bit data into voltages"""
        self.write(":WAV:YREF?")
        y_reference=float(self.read(20))
        self.write(":WAV:YOR?")
        y_origin=float(self.read(20))
        self.write(":WAV:YINC?")
        y_increment=float(self.read(20))
        #converted_data=y_reference-y_origin-data*y_increment
        converted_data=((y_reference-1)-data)*y_increment-y_origin
        return converted_data

    def get_channel_data(self, channel_number):
        """Automatically returns the y-values in volts for the specified channel"""
        self.write(":WAV:POIN:MODE RAW")
        self.write(":WAV:FORM BYTE")
        self.write(":WAVEFORM:POINTS 10240")
        self.write(":STOP")
        self.write(":WAV:DATA? CHAN%d" % channel_number)
        time.sleep(0.01)
        number_digits=self.read(2)
        number_digits=int(number_digits[1])
        number_data_points=int(self.read(number_digits))
        rawdata = self.read(number_data_points)
        data = numpy.frombuffer(rawdata, 'B')
        while len(data)<number_data_points:
            raise BufferError( "Data length should be %d, but is only %d" %
                (number_data_points,len(data)) )
            #print  ("Data length should be %d, but is only %d" %
            #        (number_data_points,len(data)))
            #time.sleep(0.05)
            #raw_data=self.read(number_data_points)
            #np.append(data,numpy.frombuffer(rawdata, 'B'))
        data=self.convert_channel_data(data)
        return data

    def get_time_data(self,channel_data):
        """Finds the times corresponding to the channel_data

        channel_data can be various lengths, hence the need to pass it as
        an argument
        This method assumes scope settings haven't been changed since the
        data was taken"""
        length=len(channel_data)
        self.write(":WAV:XINC?")
        delta_t=float(self.read(20))
        self.write(":WAV:XOR?")
        time_offset=float(self.read(20))
        time_data=numpy.linspace(0.,length,length)
        time_data=time_data*delta_t+time_offset
        return time_data

    def check_channel_scaling(self,channel_number):
        """Gets first entry as an ASCII string and prints out conversion as well"""
        converted_data=self.get_channel_data(channel_number)
        self.write("WAV:FORM ASCII")
        self.write(":WAV:DATA? CHAN%d" % channel_number)
        number_digits=self.read(2)
        number_digits=int(number_digits[1])
        number_data_points=int(self.read(number_digits))
        ascii_data= self.read(25).split(',')[0]
        print "ASCII: %s, and converted: %f" % (ascii_data, converted_data[0])
        return

    def get_single_trace(self,channel_number):
        """Takes a new trace and returns the channel_data"""
        self.write(":SINGLE")
        self.write(":TRIGger:STATus?")
        trigger_status=self.read()
        while trigger_status!='STOP\n':
            time.sleep(0.01)
            self.write(":TRIGger:STATus?")
            trigger_status=self.read()
        data=self.get_channel_data(channel_number)
        return data

    def get_multiple_traces(self,n_traces,channel_number=1):
        """Takes multiple traces and returns an array with each trace as a column"""
        one_data=self.get_single_trace(channel_number)
        all_data=numpy.zeros([n_traces+1,len(one_data)])
        all_data[1]=one_data
        for j in range(2,n_traces+1):
            one_data=self.get_single_trace(channel_number)
            all_data[j]=one_data
        time_data=self.get_time_data(one_data)
        all_data[0]=time_data
        all_data=numpy.transpose(all_data)
        return all_data


    def reset(self):
        """Reset the instrument"""
        self.meas.sendReset()


class Measurement_Series:
    """Handles the data from one series of measurements"""

    def __init__(self,scope):
        """Initializes a Measurement_Series instance from an AgilentScope"""
        self.scope=scope
        self.chanel_data=np.array([])
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

    def __getattr__(self,converted_data):
        """Converts channel_data to real voltages and returns the result"""
        converted_data=2+2
        return converted_data

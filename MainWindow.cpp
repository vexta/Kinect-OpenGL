#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Ole2.h>
#include <Kinect.h>
#include <Windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// camera rotation
#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/glut.h>

#pragma region constants
const int width = 512;
const int height = 424;
const int colorWidth = 1920;
const int colorHeight = 1080;
#define InfraredSourceValueMaximum static_cast<float>(USHRT_MAX)
#define InfraredOutputValueMinimum 0.01f 
#define InfraredOutputValueMaximum 1.0f
#define InfraredSceneValueAverage 0.08f
#define InfraredSceneStandardDeviations 3.0f
#pragma endregion
#pragma region textureData
GLuint textureId;
GLubyte colorData[colorWidth * colorHeight * 4];
GLubyte depthData[width * height * 4];
GLubyte infraredData[width * height * 4];
ColorSpacePoint depthDataColorCoordinates[width * height];
CameraSpacePoint depthDataCameraCoordinates[width * height];
#pragma endregion
#pragma region sensors
IKinectSensor* kinectSensor;
IMultiSourceFrameReader* multiSourceFrameReader;
ICoordinateMapper* coordinateMapper;
#pragma endregion
#pragma region pointCloud
GLuint vboId;
GLuint cboId;
#pragma endregion
#pragma region bodyTracking
BOOLEAN isBodyTracked;
Joint joints[JointType_Count];
#pragma endregion
#pragma region modes
enum Modes { MODE_COLOR, MODE_DEPTH, MODE_INFRARED, MODE_POINTCLOUD, MODE_SKELETON, MODE_MESH };
Modes mode = MODE_COLOR;
#pragma endregion

// perform initialization
bool initializeKinect() {
	// initialize sensor
	if (FAILED(GetDefaultKinectSensor(&kinectSensor)) || !kinectSensor) return false;
	kinectSensor->Open();
	
	// initialize coordinate mapper and multi-source frame reader
	kinectSensor->get_CoordinateMapper(&coordinateMapper);
	kinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color | FrameSourceTypes::FrameSourceTypes_Infrared | FrameSourceTypes::FrameSourceTypes_Body, &multiSourceFrameReader);

	return true;
}

void getColorData(IMultiSourceFrame* frame, GLubyte* destination, GLubyte *threeColorData) {
	// initialize frame reference
	IColorFrame* colorFrame;
	IColorFrameReference* frameReference = NULL;
	frame->get_ColorFrameReference(&frameReference);
	frameReference->AcquireFrame(&colorFrame);
	if (frameReference) frameReference->Release();

	if (!colorFrame) return;

	// fill data from frame
	colorFrame->CopyConvertedFrameDataToArray(colorWidth * colorHeight * 4, threeColorData, ColorImageFormat_Rgba);

	// write color array for vertices
	float* floatDestination = (float*)destination;
	for (int i = 0; i < width * height; i++) {
		ColorSpacePoint point = depthDataColorCoordinates[i];
		// check if color pixel coordinates are out of bounds and fill them with black pixels
		if (point.X < 0 || point.Y < 0 || point.X > colorWidth || point.Y > colorHeight) {
			*floatDestination++ = 0;
			*floatDestination++ = 0;
			*floatDestination++ = 0;
		}
		else {
			int idx = (int)point.X + colorWidth*(int)point.Y;
			*floatDestination++ = threeColorData[4 * idx + 0] / 255.;
			*floatDestination++ = threeColorData[4 * idx + 1] / 255.;
			*floatDestination++ = threeColorData[4 * idx + 2] / 255.;
		}
	}

	if (colorFrame) colorFrame->Release();
}

void getDepthData(IMultiSourceFrame* frame, GLubyte* destination, GLubyte* targetDepthData, CameraSpacePoint *cameraPoints, ColorSpacePoint *colorPoints) {
	// initialize frame reference
	IDepthFrame* depthFrame;
	IDepthFrameReference* frameReference = NULL;
	frame->get_DepthFrameReference(&frameReference);
	frameReference->AcquireFrame(&depthFrame);
	if (frameReference) frameReference->Release();

	if (!depthFrame) return;

	// fill data from frame
	unsigned int capacity;
	unsigned short* buffer;
	depthFrame->AccessUnderlyingBuffer(&capacity, &buffer);

	
	USHORT minDepthReliableDistance, maxDepthReliableDistance;
	depthFrame->get_DepthMinReliableDistance(&minDepthReliableDistance);
	depthFrame->get_DepthMaxReliableDistance(&maxDepthReliableDistance);

	const unsigned short* currentData = buffer;
	const unsigned short* dataEnd = buffer + (width*height);

	while (currentData < dataEnd)
	{
		USHORT depth = *currentData;
		BYTE intensity = static_cast<BYTE>((depth >= minDepthReliableDistance) && (depth <= USHRT_MAX) ? (depth % 256) : 0);

		*targetDepthData++ = intensity;
		*targetDepthData++ = intensity;
		*targetDepthData++ = intensity;
		*targetDepthData++ = 0xFF;
		++currentData;
	}

	// write vertex coordinates
	coordinateMapper->MapDepthFrameToCameraSpace(width * height, buffer, width * height, cameraPoints);
	float* floatDestination = (float*)destination;
	for (int i = 0; i < capacity; i++) {
		*floatDestination++ = cameraPoints[i].X;
		*floatDestination++ = cameraPoints[i].Y;
		*floatDestination++ = cameraPoints[i].Z;
	}

	// fill in depth to color map
	coordinateMapper->MapDepthFrameToColorSpace(width * height, buffer, width * height, colorPoints);
	if (depthFrame) depthFrame->Release();
}

void getInfraredData(IMultiSourceFrame* frame, GLubyte* destination) {
	// initialize frame reference
	IInfraredFrame* infraredFrame;
	IInfraredFrameReference* frameReference = NULL;
	frame->get_InfraredFrameReference(&frameReference);
	frameReference->AcquireFrame(&infraredFrame);
	if (frameReference) frameReference->Release();

	if (!infraredFrame) return;

	// fill data from frame
	unsigned int capacity;
	unsigned short* buffer;
	infraredFrame->AccessUnderlyingBuffer(&capacity, &buffer);

	// copy grayscale data to GLubyte structure
	const unsigned short* dataEnd = buffer + (width*height);

	while (buffer < dataEnd)
	{
		// normalize the incoming infrared data (ushort) to a float ranging from 
		// [InfraredOutputValueMinimum, InfraredOutputValueMaximum] by
		// 1. dividing the incoming value by the source maximum value
		float intensityRatio = static_cast<float>(*buffer) / InfraredSourceValueMaximum;

		//// 2. dividing by the (average scene value * standard deviations)
		intensityRatio /= InfraredSceneValueAverage * InfraredSceneStandardDeviations;

		//// 3. limiting the value to InfraredOutputValueMaximum
		intensityRatio = min(InfraredOutputValueMaximum, intensityRatio);

		//// 4. limiting the lower value InfraredOutputValueMinimym
	 	intensityRatio = max(InfraredOutputValueMinimum, intensityRatio);

		// 5. converting the normalized value to a byte and using the result
		// as the RGB components required by the image
		byte intensity = *buffer;//static_cast<byte>(intensityRatio * 255.0f);
		*destination++ = intensity;
		*destination++ = intensity;
		*destination++ = intensity;
		*destination++ = 0xff;
		++buffer;
	}

	if (infraredFrame) infraredFrame->Release();
}

void getBodyData(IMultiSourceFrame* frame) {
	IBodyFrame* bodyFrame;
	IBodyFrameReference* frameReference = NULL;
	frame->get_BodyFrameReference(&frameReference);
	frameReference->AcquireFrame(&bodyFrame);
	if (frameReference) frameReference->Release();

	if (!bodyFrame) return;

	IBody* body[BODY_COUNT] = { 0 };
	bodyFrame->GetAndRefreshBodyData(BODY_COUNT, body);
	for (int i = 0; i < BODY_COUNT; i++) {
		body[i]->get_IsTracked(&isBodyTracked);
		if (isBodyTracked) {
			body[i]->GetJoints(JointType_Count, joints);
			break;
		}
	}

	if (bodyFrame) bodyFrame->Release();
}

void getKinectData() {
	IMultiSourceFrame* frame = NULL;
	if (SUCCEEDED(multiSourceFrameReader->AcquireLatestFrame(&frame))) {
		GLubyte* ptr;
		glBindBuffer(GL_ARRAY_BUFFER, vboId);
		ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		if (ptr) {
			getDepthData(frame, ptr, depthData, depthDataCameraCoordinates, depthDataColorCoordinates);
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, cboId);
		ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		if (ptr) {
			getColorData(frame, ptr, colorData);
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		getBodyData(frame);
		getInfraredData(frame, infraredData);
	}
	if (frame) frame->Release();
}

void rotateCamera() {
	static double angle = 0.;
	static double radius = 3.;
	double x = radius*sin(angle);
	double z = radius*(1 - cos(angle)) - radius / 2;
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(x, 0, z, 0, 0, radius / 2, 0, 1, 0);
	angle += 0.001;
}

void drawKinectData() {
	getKinectData();
	glClearColor(0, 0, 0, 0);
	if (mode == MODE_COLOR || mode == MODE_INFRARED || mode == MODE_DEPTH)
	{
		glBindTexture(GL_TEXTURE_2D, textureId);

		if (mode == MODE_COLOR)
		{
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, colorWidth, colorHeight, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)colorData);
		}
		else if (mode == MODE_DEPTH)
		{
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)depthData);
		}
		else glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)infraredData);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f);
		glVertex3f(0, 0, 0);
		glTexCoord2f(1.0f, 0.0f);
		glVertex3f(colorWidth, 0, 0);
		glTexCoord2f(1.0f, 1.0f);
		glVertex3f(colorWidth, colorHeight, 0.0f);
		glTexCoord2f(0.0f, 1.0f);
		glVertex3f(0, colorHeight, 0.0f);
		glEnd();
	}
	else if (mode == MODE_POINTCLOUD)
	{
		glViewport(0, 0, width, height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, vboId);
		glVertexPointer(3, GL_FLOAT, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, cboId);
		glColorPointer(3, GL_FLOAT, 0, NULL);

		glPointSize(1.f);
		glDrawArrays(GL_POINTS, 0, width*height);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);

		glViewport(0, 0, colorWidth, colorHeight);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
	}
	else if (mode == MODE_SKELETON)
	{
		if (isBodyTracked) {
			// Draw some arms
			CameraSpacePoint& lh = joints[JointType_WristLeft].Position;
			const CameraSpacePoint& le = joints[JointType_ElbowLeft].Position;
			const CameraSpacePoint& ls = joints[JointType_ShoulderLeft].Position;
			const CameraSpacePoint& rh = joints[JointType_WristRight].Position;
			const CameraSpacePoint& re = joints[JointType_ElbowRight].Position;
			const CameraSpacePoint& rs = joints[JointType_ShoulderRight].Position;
			glBegin(GL_LINES);
			glColor3f(1.f, 0.f, 0.f);
			glVertex3f(lh.X, lh.Y, lh.Z);
			glVertex3f(le.X, le.Y, le.Z);
			glVertex3f(le.X, le.Y, le.Z);
			glVertex3f(ls.X, ls.Y, ls.Z);
			glVertex3f(rh.X, rh.Y, rh.Z);
			glVertex3f(re.X, re.Y, re.Z);
			glVertex3f(re.X, re.Y, re.Z);
			glVertex3f(rs.X, rs.Y, rs.Z);
			glEnd();
		}
	}
}

static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
	if (key == GLFW_KEY_1 && action == GLFW_PRESS)
		mode = MODE_COLOR;
	if (key == GLFW_KEY_2 && action == GLFW_PRESS)
		mode = MODE_DEPTH;
	if (key == GLFW_KEY_3 && action == GLFW_PRESS)
		mode = MODE_INFRARED;
	if (key == GLFW_KEY_4 && action == GLFW_PRESS)
		mode = MODE_POINTCLOUD;
	if (key == GLFW_KEY_5 && action == GLFW_PRESS)
		mode = MODE_SKELETON;
	if (key == GLFW_KEY_6 && action == GLFW_PRESS)
		// mode = MODE_MESH;
		void;
}

int main(void)
{
	GLFWwindow* window;

	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) exit(EXIT_FAILURE);
	if (!initializeKinect()) exit(EXIT_FAILURE);

	window = glfwCreateWindow(colorWidth, colorHeight, "Kinect GLFW Prototype", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);
	glewInit();
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, key_callback);

	// Initialize textures
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, colorWidth, colorHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)colorData);
	glBindTexture(GL_TEXTURE_2D, 0);

	// OpenGL setup
	glClearColor(0, 0, 0, 0);
	glClearDepth(1.0f);
	glEnable(GL_TEXTURE_2D);

	// Set up array buffers
	glGenBuffers(1, &vboId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, width * height * 3 * 4, 0, GL_DYNAMIC_DRAW);
	glGenBuffers(1, &cboId);
	glBindBuffer(GL_ARRAY_BUFFER, cboId);
	glBufferData(GL_ARRAY_BUFFER, width * height * 3 * 4, 0, GL_DYNAMIC_DRAW);

	// Camera setup
	glViewport(0, 0, colorWidth, colorHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, colorWidth, colorHeight, 0, 1, -1);
	glMatrixMode(GL_MODELVIEW);

	while (!glfwWindowShouldClose(window))
	{
		float ratio;
		int width, height;

		glfwGetFramebufferSize(window, &width, &height);
		drawKinectData();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	glfwTerminate();
	exit(EXIT_SUCCESS);
}

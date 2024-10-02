from manim import *
import numpy as np

class GradientDescent(Scene):
    def construct(self):
        # Setup axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 16, 2],
            axis_config={"color": BLUE},
        )
        # Label axes
        labels = axes.get_axis_labels(x_label="x", y_label="f(x)")

        # Function to optimize f(x) = x^2
        graph = axes.plot(lambda x: x**2+2, color=WHITE)
        #graph_label = axes.get_graph_label(graph, label='f(x)')

        # Initial point
        x_value = -3
        point = Dot(color=RED).move_to(axes.c2p(x_value, x_value**2+2))
        point_label = MathTex(f"x = {x_value:.2f}", color=RED).next_to(point, UP)

        # Gradient descent formula on screen
        eta = 0.25  # Learning rate
        formula = MathTex(r"x_n = x_{n-1} - \eta \cdot f'(x_{n-1})", font_size=50)
        formula.move_to(2*UP)

        # Adding components to the scene
        self.add(axes, labels, graph, point, point_label, formula)

        # Gradient descent iterations
        for _ in range(6):
            old_x_value = x_value
            gradient = 2 * (x_value)  # Derivative of x^2
            x_value -= eta * gradient  # Update rule

            # Update point and label
            new_point = Dot(color=RED).move_to(axes.c2p(x_value, x_value**2+2))
            new_point_label = MathTex(f"x = {x_value:.2f}", color=RED).next_to(new_point, UP)

            # Draw tangent line at the old point
            tangent_line = always_redraw(lambda: self.get_tangent_line(axes, old_x_value))
            self.add(tangent_line)

            # Play animation
            self.play(
                Transform(point, new_point),
                Transform(point_label, new_point_label),
                run_time=1
            )
            self.remove(tangent_line)

            self.wait(0.5)  # Wait half a second between updates

        self.wait(1)  # Wait at the end of the animation

    def f(self, x):
        return x**2 + 2
    def df(self,x):
        return 2*x
    
    def get_tangent_line(self, axes, x_value):
        slope = 2 * x_value  # Derivative at x
        intercept = x_value**2 + 2 - slope * x_value
        left = axes.x_range[0]
        right = axes.x_range[1]
        return Line(
            start=axes.c2p(left, slope * left + intercept),
            end=axes.c2p(right, slope * right + intercept),
            color=YELLOW,
            stroke_width=4,
        )
    
class GradientDescent3D(ThreeDScene):
    def construct(self):
        # Setting up axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 10, 2],
            x_length=7,
            y_length=7,
            z_length=5,
            axis_config={"include_tip": True},
        )

        # Define the 3D paraboloid cost function (Convex function)
        def cost_function(x, y):
            return 0.5*(x**2 + y**2) -2

        # Create a surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, cost_function(u, v)),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(40, 40),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        # Initial guess point
        point = Dot3D(axes.c2p(2.5, 2.5, cost_function(2.5, 2.5)), color=RED)
        x_value_label = DecimalNumber(2.5, num_decimal_places=2).set_color(WHITE)
        y_value_label = DecimalNumber(2.5, num_decimal_places=2).set_color(WHITE)
        z_value_label = DecimalNumber(cost_function(2.5, 2.5), num_decimal_places=2).set_color(WHITE)
        
        value_label_group = VGroup(x_value_label,y_value_label,z_value_label)

        x_label = MathTex("x=").next_to(x_value_label, LEFT)
        y_label = MathTex("y=").next_to(y_value_label, LEFT)
        z_label = MathTex("f(x, y)=").next_to(z_value_label, LEFT)

        # Group labels together for convenient positioning
        labels_group = VGroup(x_label, x_value_label, y_label, y_value_label, z_label, z_value_label)
        labels_group.arrange(DOWN).to_corner(UL)

        # Add elements to the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.add(axes, surface, point)
        self.add_fixed_in_frame_mobjects(labels_group)  # Ensures the labels stay fixed to the screen
        self.add_fixed_in_frame_mobjects(value_label_group)

        # Set up continuous camera rotation
        self.begin_ambient_camera_rotation(rate=0.1)

        # Gradient descent parameters
        steps = 20
        alpha = 0.15  # Learning rate
        point_pos = np.array([2.5, 2.5])

        for _ in range(steps):
            # Calculate gradient at the current position
            grad_x = 2 * point_pos[0]
            grad_y = 2 * point_pos[1]

            # Update point position using gradient descent
            new_point_pos = point_pos - alpha * np.array([grad_x, grad_y])
            new_z = cost_function(new_point_pos[0], new_point_pos[1])

            # Update the position of the point
            new_point = Dot3D(axes.c2p(new_point_pos[0], new_point_pos[1], new_z), color=RED)
            self.play(Transform(point, new_point), run_time=0.5)

            # Update labels with the new values
            x_value_label.set_value(new_point_pos[0])
            y_value_label.set_value(new_point_pos[1])
            z_value_label.set_value(new_z)

            self.wait(0.2)  # Pause for visibility

            # Update the point's position for the next iteration
            point_pos = new_point_pos
        # Final camera movement to flip the graph upside down
        # Flip the graph to show it's at the minimum
        self.move_camera(phi=180 * DEGREES, theta=0 * DEGREES, run_time=3)

        self.wait(2)  # Pause before ending the scene


class GradientDescentVisualization(ThreeDScene):
    def construct(self):
        # Setting up the axes
        axes = ThreeDAxes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            z_range=[-2, 3, 1],
            x_length=7,
            y_length=7,
            z_length=5,
            axis_config={"include_tip": True},
        )

        # Axis labels
        theta_0_label = MathTex(r"\theta_0").next_to(axes.c2p(1, 0, 0), DOWN)
        theta_1_label = MathTex(r"\theta_1").next_to(axes.c2p(0, 1, 0), LEFT)
        cost_label = MathTex(r"J(\theta_0, \theta_1)").rotate(PI / 2).next_to(axes.c2p(0, 0, 3), LEFT)

        # Define the cost function
        def cost_function(x, y):
            return np.sin(3 * np.pi * x) * np.cos(3 * np.pi * y)

        # Create the 3D surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, cost_function(u, v)),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE, YELLOW],
            stroke_color=WHITE,
            stroke_width=0.5
        )

        # Gradient descent path
        steps = 12
        alpha = 0.1  # Learning rate
        start_pos = np.array([0.9, 0.9])  # Starting point for the descent
        descent_path = [start_pos]

        # Perform gradient descent steps
        for _ in range(steps):
            grad_x = 3 * np.pi * np.cos(3 * np.pi * start_pos[0]) * np.cos(3 * np.pi * start_pos[1])
            grad_y = -3 * np.pi * np.sin(3 * np.pi * start_pos[0]) * np.sin(3 * np.pi * start_pos[1])
            new_pos = start_pos - alpha * np.array([grad_x, grad_y])
            descent_path.append(new_pos)
            start_pos = new_pos

        # Create path and arrows
        descent_dots = []
        for i in range(len(descent_path) - 1):
            dot = Dot3D(
                point=axes.c2p(descent_path[i][0], descent_path[i][1], cost_function(descent_path[i][0], descent_path[i][1])),
                color=BLACK
            )
            descent_dots.append(dot)
            # Add an arrow to indicate the direction of descent
            arrow = Line3D(
                start=axes.c2p(descent_path[i][0], descent_path[i][1], cost_function(descent_path[i][0], descent_path[i][1])),
                end=axes.c2p(descent_path[i+1][0], descent_path[i+1][1], cost_function(descent_path[i+1][0], descent_path[i+1][1])),
                color=BLACK,
                stroke_width=2
            )
            self.add(arrow)

        # Add everything to the scene
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.add(axes, surface, theta_0_label, theta_1_label, cost_label)
        
        # Add the descent dots
        self.add(*descent_dots)
        
        # Rotate the camera to better visualize the surface
        self.move_camera(phi=75 * DEGREES, theta=-60 * DEGREES, run_time=4)
        
        # Pause to show the result
        self.wait(3)

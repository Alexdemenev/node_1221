from inspect import cleandoc
import torch
import torch.nn.functional as F

MAX_RESOLUTION = 4096


class Example:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("Image", {"tooltip": "This is an image"}),
                "int_field": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 64,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "float_field": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "number",
                    },
                ),
                "print_to_screen": (["enable", "disable"],),
                "string_field": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "test"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(
                f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """
            )
        # do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


class ImageSelector:
    CATEGORY = "custom_1221"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "choose_image"

    def choose_image(self, images):
        brightness = list(torch.mean(image.flatten()).item() for image in images)
        brightest = brightness.index(max(brightness))
        result = images[brightest].unsqueeze(0)
        return (result,)


class ImageComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": (
                    "INT",
                    {
                        "default": 0,
                        "min": -MAX_RESOLUTION,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "y": (
                    "INT",
                    {
                        "default": 0,
                        "min": -MAX_RESOLUTION,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "offset_x": (
                    "INT",
                    {
                        "default": 0,
                        "min": -MAX_RESOLUTION,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "offset_y": (
                    "INT",
                    {
                        "default": 0,
                        "min": -MAX_RESOLUTION,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
                "auto_offset": (["enable", "disable"],),
                "auto_offset_x": (["enable", "disable"],),
                "auto_offset_y": (["enable", "disable"],),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "custom_1221"

    def execute(
        self,
        destination,
        source,
        x,
        y,
        offset_x,
        offset_y,
        auto_offset="disable",
        auto_offset_x="disable",
        auto_offset_y="disable",
        mask=None,
    ):
        if mask is None:
            mask = torch.ones_like(source)[:, :, :, 0]

        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        if mask.shape[1:3] != source.shape[1:3]:
            mask = F.interpolate(
                mask.permute([0, 3, 1, 2]),
                size=(source.shape[1], source.shape[2]),
                mode="bicubic",
            )
            mask = mask.permute([0, 2, 3, 1])

        if mask.shape[0] > source.shape[0]:
            mask = mask[: source.shape[0]]
        elif mask.shape[0] < source.shape[0]:
            mask = torch.cat(
                (mask, mask[-1:].repeat((source.shape[0] - mask.shape[0], 1, 1, 1))),
                dim=0,
            )

        if destination.shape[0] > source.shape[0]:
            destination = destination[: source.shape[0]]
        elif destination.shape[0] < source.shape[0]:
            destination = torch.cat(
                (
                    destination,
                    destination[-1:].repeat(
                        (source.shape[0] - destination.shape[0], 1, 1, 1)
                    ),
                ),
                dim=0,
            )

        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]

        if len(x) < destination.shape[0]:
            x = x + [x[-1]] * (destination.shape[0] - len(x))
        if len(y) < destination.shape[0]:
            y = y + [y[-1]] * (destination.shape[0] - len(y))

        if auto_offset == "enable":
            # Центрируем source относительно destination для каждого изображения в батче
            x = [
                (destination.shape[2] - source.shape[2]) // 2
                for _ in range(destination.shape[0])
            ]
            y = [
                (destination.shape[1] - source.shape[1]) // 2
                for _ in range(destination.shape[0])
            ]
        if auto_offset_x == "enable":
            x = [
                (destination.shape[2] - source.shape[2]) // 2
                for _ in range(destination.shape[0])
            ]
        if auto_offset_y == "enable":
            y = [
                (destination.shape[1] - source.shape[1]) // 2
                for _ in range(destination.shape[0])
            ]

        if not isinstance(offset_x, list):
            offset_x = [offset_x] * len(x)
        if not isinstance(offset_y, list):
            offset_y = [offset_y] * len(y)

        x = [xi + ox for xi, ox in zip(x, offset_x)]
        y = [yi + oy for yi, oy in zip(y, offset_y)]

        output = []
        for i in range(destination.shape[0]):
            d = destination[i].clone()
            s = source[i]
            m = mask[i]

            if x[i] + source.shape[2] > destination.shape[2]:
                s = s[:, :, : destination.shape[2] - x[i], :]
                m = m[:, :, : destination.shape[2] - x[i], :]
            if y[i] + source.shape[1] > destination.shape[1]:
                s = s[:, : destination.shape[1] - y[i], :, :]
                m = m[: destination.shape[1] - y[i], :, :]

            # output.append(s * m + d[y[i]:y[i]+s.shape[0], x[i]:x[i]+s.shape[1], :] * (1 - m))
            d[y[i] : y[i] + s.shape[0], x[i] : x[i] + s.shape[1], :] = s * m + d[
                y[i] : y[i] + s.shape[0], x[i] : x[i] + s.shape[1], :
            ] * (1 - m)
            output.append(d)

        output = torch.stack(output)

        # apply the source to the destination at XY position using the mask
        # for i in range(destination.shape[0]):
        #    output[i, y[i]:y[i]+source.shape[1], x[i]:x[i]+source.shape[2], :] = source * mask + destination[i, y[i]:y[i]+source.shape[1], x[i]:x[i]+source.shape[2], :] * (1 - mask)

        # for x_, y_ in zip(x, y):
        #    output[:, y_:y_+source.shape[1], x_:x_+source.shape[2], :] = source * mask + destination[:, y_:y_+source.shape[1], x_:x_+source.shape[2], :] * (1 - mask)

        # output[:, y:y+source.shape[1], x:x+source.shape[2], :] = source * mask + destination[:, y:y+source.shape[1], x:x+source.shape[2], :] * (1 - mask)
        # output = destination * (1 - mask) + source * mask

        return (output,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Example": Example,
    "Image Selector": ImageSelector,
    "Image Composite": ImageComposite,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Example Node",
    "Image Selector": "Image Selector",
    "Image Composite": "Image Composite",
}
